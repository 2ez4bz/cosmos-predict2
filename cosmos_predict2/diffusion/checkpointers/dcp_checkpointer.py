import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
import enum
import functools
import multiprocessing
import os
import time
import attrs
from collections import namedtuple
from multiprocessing import get_context
from typing import Any, Dict, List, Set, Tuple, Union

import torch.distributed
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from cosmos_predict2.utils import log
from cosmos_predict2.utils.config import CheckpointConfig as BaseCheckpointConfig
from cosmos_predict2.utils.config import make_freezable
from cosmos_predict2.utils import callback, distributed, log, misc
from cosmos_predict2.utils.config import CheckpointConfig, JobConfig
from cosmos_predict2.utils.easy_io import easy_io
from cosmos_predict2.utils.model import Model

@make_freezable
@attrs.define(slots=False)
class CheckpointConfig(BaseCheckpointConfig):
    load_ema_to_reg: bool = False

class AbstractCheckpointer(ABC):
    """The checkpointer class. Supports checkpoint saving/loading to both local disk or object store."""

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        self.config_checkpoint = config_checkpoint
        # Set the callback functions.
        self.callbacks = callbacks

        # Set checkpoint directories for local and object store paths
        self._local_dirname = os.path.join(config_job.path_local, "checkpoints")
        self._object_store_dirname = os.path.join(config_job.path, "checkpoints")

        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None
        self.verbose = config_checkpoint.verbose
        self.keys_not_to_resume = config_checkpoint.keys_not_to_resume
        self.broadcast_via_filesystem = config_checkpoint.broadcast_via_filesystem
        # Create the object store client interface.
        self.load_s3_backend_key = None
        self.save_s3_backend_key = None

    @abstractmethod
    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        pass

    @abstractmethod
    def load(
        self,
        model: Model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        pass

    @property
    def save_bucket(self):
        """Get the bucket name for saving checkpoints."""
        return self.config_checkpoint.save_to_object_store.bucket if self.save_to_object_store else None

    @property
    def load_bucket(self):
        """Get the bucket name for loading checkpoints."""
        return self.config_checkpoint.load_from_object_store.bucket if self.load_from_object_store else None

    @property
    def save_dirname(self):
        return (
            f"s3://{self.save_bucket}/{self._object_store_dirname}"
            if self.save_to_object_store
            else self._local_dirname
        )

    @property
    def load_dirname(self):
        return self._local_dirname

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        checkpoint_path = os.path.join(self.load_dirname, "latest_checkpoint.txt")
        if easy_io.exists(f"{checkpoint_path}", backend_key=self.load_s3_backend_key):
            checkpoint_file = easy_io.load(f"{checkpoint_path}", backend_key=self.load_s3_backend_key).strip()

        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        checkpoint_path = os.path.join(self.save_dirname, "latest_checkpoint.txt")
        easy_io.dump(
            content,
            checkpoint_path,
            backend_key=self.save_s3_backend_key,
        )

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        return
        if not easy_io.exists(f"{checkpoint_path}", backend_key=self.load_s3_backend_key):
            raise FileNotFoundError(f"File not found (object store): {checkpoint_path}")



StateDictItemPath = namedtuple("StateDictItemPath", ["state_dict", "save_path"])

# (qsh 2025-01-01) the design is from https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/torchtitan/checkpoint.py
# we recreate wrapper when needed instead of creating one from the beginning.
# to people who find it difficult to digest the code, official tutorial for torch dcp may be helpful


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: Union[nn.Module, List[nn.Module]], load_ema_to_reg: bool = False):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.load_ema_to_reg = load_ema_to_reg
        if self.load_ema_to_reg:
            assert isinstance(model, Model), "ModelWrapper only supports DiffusionModel when load_ema_to_reg is True"

    def state_dict(self) -> Dict[str, Any]:
        _state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}
        if self.load_ema_to_reg:
            assert not self.model[
                0
            ].config.ema.enabled, "EMA is enabled, can not load EMA weights to regular model weights"
            all_keys = list(_state_dict.keys())
            assert all(k.startswith("net.") for k in all_keys), "All keys must start with net."
            for k in all_keys:
                _state_dict[k.replace("net.", "net_ema.")] = _state_dict.pop(k)
        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.load_ema_to_reg:
            assert not self.model[
                0
            ].config.ema.enabled, "EMA is enabled, can not load EMA weights to regular model weights"
            all_keys = list(state_dict.keys())
            assert all(k.startswith("net_ema.") for k in all_keys), "All keys must start with net_ema."
            for k in all_keys:
                state_dict[k.replace("net_ema.", "net.")] = state_dict.pop(k)
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    pass


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config: CheckpointConfig,
    job_config: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(checkpoint_config, job_config, None, disable_async=True)

    try:
        while True:
            log.debug("Checkpoint background process is ready for next task")
            sender_queue.put(SaveDone())

            log.debug("Waiting to receive new state_dict")
            received_data = receiver_queue.get()
            log.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                log.info("Received termination signal for checkpoint background process")
                return

            assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)

            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds")

    finally:
        log.info("Cleaning up: destroying distributed process group")
        torch.distributed.destroy_process_group()


class DistributedCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: callback.CallBackGroup,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> Tuple[Set, Union[str, None]]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            print(f"{(self.load_dirname, latest_checkpoint_file)=}")
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]
        return set(resume_keys), checkpoint_path

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        # TODO: (qsh 2025-01-01) the current interface design can not load callback states.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=DefaultLoadPlanner(allow_partial_load=True),
                    )
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _optim_wrapper = OptimizerWrapper(model, optimizer)
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=DefaultLoadPlanner(allow_partial_load=True),
                    )
                    _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=DefaultLoadPlanner(allow_partial_load=True),
                    )
                    scheduler.load_state_dict(_state_dict)
                elif key == "trainer":
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=DefaultLoadPlanner(allow_partial_load=True),
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model)
        return iteration

    def _async_with_pinned_memory(self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]) -> None:
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            log.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file

        # TODO: (qsh 2025-01-02) a better name?
        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))

            sync_func()
            self.staging = False

    def get_storage_writer(self, checkpoint_path: str) -> FileSystemWriter:
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> FileSystemReader:
        return FileSystemReader(checkpoint_path)

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            storage_writer = self.get_storage_writer(full_checkpoint_path)
            dcp.save(
                v,
                storage_writer=storage_writer,
                planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
            )

        self._write_latest_checkpoint_file(checkpoint_file)
        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (Model): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": OptimizerWrapper(model, optimizer).state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            self.save_state_dict_worker(to_save_dict, checkpoint_file)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
