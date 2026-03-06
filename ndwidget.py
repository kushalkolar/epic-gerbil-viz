# import glfw
import numpy as np
from functools import partial
import soundfile as sf
from scipy.signal import spectrogram
from decord import VideoReader, gpu
import fastplotlib as fpl
from ipywidgets import VBox, HBox
import os
from typing import *
from abc import ABC, abstractmethod
from pathlib import Path
from warnings import warn

# Some stuff I copied from mesmerize-core for lazy-loading the video files using decord

slice_or_int_or_range = Union[int, slice, range]

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"

class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """

    def __array__(self, dtype=None, copy=None):
        if copy:
            return copy(self)

        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        str
            data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        """
        float
            min value of the array if it were fully computed
        """
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        """
        float
            max value of the array if it were fully computed
        """
        pass

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

    @property
    def nbytes_gb(self) -> float:
        """
        float
            number of gigabytes for the array if it were fully computed
        """
        return self.nbytes / 1e9

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here. Computes the array at the desired indices.

        Parameters
        ----------
        indices: Union[int, slice]
            the user's desired slice, i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        pass

    def __getitem__(self, item: Union[int, Tuple[slice_or_int_or_range]]):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item

        elif isinstance(item, tuple):
            if item[-1] is Ellipsis:
                item = item[:-1]

            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            return self._compute_at_indices(item[0])

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = (slice(start, stop, step),)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, (int, np.integer)):
            return self._compute_at_indices(indexer)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} @{hex(id(self))}\n"
            f"{self.__class__.__doc__}\n"
            f"Frames are computed only upon indexing\n"
            f"shape [frames, x, y]: {self.shape}\n"
        )


class LazyVideo(LazyArray):
    def __init__(
        self,
        path: Union[Path, str],
        min_max: Tuple[int, int] = None,
        **kwargs,
    ):
        """
        LazyVideo reader, basically just a wrapper for ``decord.VideoReader``.
        Should support opening anything that decord can open.

        **Important:** requires ``decord`` to be installed: https://github.com/dmlc/decord

        Parameters
        ----------
        path: Path or str
            path to video file

        min_max: Tuple[int, int], optional
            min and max vals of the entire video, uses min and max of 10th frame if not provided

        as_grayscale: bool, optional
            return grayscale frames upon slicing

        rgb_weights: Tuple[float, float, float], optional
            (r, g, b) weights used for grayscale conversion if ``as_graycale`` is ``True``.
            default is (0.299, 0.587, 0.114)

        kwargs
            passed to ``decord.VideoReader``

        Examples
        --------

        Lazy loading with CPU

        .. code-block:: python

            from mesmerize_core.arrays import LazyVideo

            vid = LazyVideo("path/to/video.mp4")

            # use fpl to visualize

            import fastplotlib as fpl

            iw = fpl.ImageWidget(vid)
            iw.show()


        Lazy loading with GPU, decord must be compiled with CUDA options to use this

        .. code-block:: python

            from decord import gpu
            from mesmerize_core.arrays import LazyVideo

            gpu_context = gpu(0)

            vid = LazyVideo("path/to/video.mp4", ctx=gpu_context)

        """
        self._video_reader = VideoReader(str(path), **kwargs)

        try:
            frame0 = self._video_reader[10].asnumpy()
            self._video_reader.seek(0)
        except IndexError:
            frame0 = self._video_reader[0].asnumpy()
            self._video_reader.seek(0)

        self._shape = (self._video_reader._num_frame, *frame0.shape)

        self._dtype = frame0.dtype

        if min_max is not None:
            self._min, self._max = min_max
        else:
            self._min = frame0.min()
            self._max = frame0.max()

        self._cache = dict()

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """[n_frames, x, y, 3 | 4]"""
        return self._shape

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self._min

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self._max

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        if isinstance(indices, slice):
            s = indices
            k = (s.start, s.stop, s.step)
            if k in self._cache.keys():
                return self._cache[k]

        a = self._video_reader[indices].asnumpy()
        self._video_reader.seek(0)

        if isinstance(indices, slice):
            s = indices
            k = (s.start, s.stop, s.step)
            self._cache[k] = a

        while len(self._cache) > 128:
            self._cache.pop(self._cache.keys()[0])

        return a


# ==== Choose ====
exp_num = 518
file_num = 0

# channel_numbers = [1, 0, 4, 5]  # Choose 4 channels   [2,0,4,5]
channel_numbers = [4, 2, 1, 0]  # Choose 4 channels   [2,0,4,5]

spec_height = 300
spec_width = 1500
vid_height = 600

# === spectrogram variables ===
fps_audio = 125000
fps_video = 29.9976  # 30
n_fft = 512
n_samples_bin = 512
n_samples_overlap = 256
window_width_sec = 5  # choose


# Functions
def make_specgram(audio, fps=125000):
    f, t, spec = spectrogram(
        audio,
        fs=fps,
        nfft=n_fft,
        nperseg=n_samples_bin,
        noverlap=n_samples_overlap,
        return_onesided=True,
    )
    # Remove the 0-frequency bin and flip the frequency axis so that high frequencies are at the top.
    f = f[1:][::-1]
    spec = np.flip(spec[1:], axis=0)
    spec = np.log(np.abs(spec) + 1e-12)
    spec32 = np.zeros(spec.shape, dtype=np.float32)
    spec32[:] = spec.astype(np.float32)
    return t, f, spec32


# === Video file naming rule based on channel number ===
video_prefix_map = {
    2: "video_center_",
    3: "video_center_",
    4: "video_gily_center_",  # change back
    5: "video_gily_center_",  # change back
    0: "video_nest_top_",  # "video_nest_top_", video_nest_side_
    1: "video_burrow_side_",  # "video_burrow_top_" video_burrow_side_
}

# maps int -> location
name_mapping = {
    2: "center-1",
    3: "center-1",  # 1
    4: "center-2",
    5: "center-2",
    0: "nest",
    1: "burrow",
}

# location_order = ["center-2","center-1", "burrow", "nest"] # ["center-2", "center-1", "burrow", "nest"]
location_order = ["center-2", "center-1", "burrow", "nest"]

base_path_audio = base_path_video = "/home/kushal/data/gerbils/example_sess/"

# === Collect paths ===
# maps location str -> path str
video_paths: dict[str, str] = dict()
audio_paths: dict[str, str] = dict()

file_num_str = f"{file_num:03d}"
print(file_num_str)
for ch in channel_numbers:
    video_prefix = video_prefix_map[ch]
    if video_prefix is None:
        print(f"Warning: Invalid channel number {ch}, skipping.")
        continue

    video_path = os.path.join(
        base_path_video, f"{video_prefix}{file_num_str}.mp4"
    )  #####
    print(video_path)
    audio_path = os.path.join(
        base_path_audio, f"channel_{ch:02d}_file_{file_num_str}.wav"
    )

    video_paths[name_mapping[ch]] = video_path
    audio_paths[name_mapping[ch]] = audio_path

# === Output the results ===
# print("Video paths:")
# for path in video_paths:
#     print(" ", path)

# print("\nAudio paths:")
# for path in audio_paths:
#     print(" ", path)


## Load video and audio ##
# # maps location name -> LazyVideo
movies: dict[str, LazyVideo] = dict()

# # maps location name -> full spectrogram
specs: dict[str, np.ndarray] = dict()


# --- Video loading ---
for location, path in video_paths.items():
    # --- check video file lengths and sizes before loading ---
    print("\n=== Checking video file info ===")
    try:
        vid = LazyVideo(path)
        n_frames = vid.shape[0]
        height, width = vid._video_reader[0].shape[:2]
        fps = vid._video_reader.get_avg_fps()
        movies[location] = (vid, fps)
        vid[0]  # MUST do this to clear the RAM because decord has a memory leak otherwise!!
        print(
            f"{os.path.basename(path)}: {n_frames} frames, {width}x{height}, {fps:.2f} fps"
        )
    except Exception as e:
        print(f"❌ Could not open {path}: {e}")

print("=== Done checking video file info ===\n")


for location, path in audio_paths.items():
    audio_data, fps_audio = sf.read(path, dtype="float32")
    print(f"loaded: {path}")

    # Unpack the outputs of make_specgram
    t_spec, f_spec, spec_data = make_specgram(audio_data, fps_audio)

    spec_reshaped = np.dstack([np.broadcast_to(t_spec[None, :], spec_data.shape), spec_data])

    specs[location] = (spec_reshaped, t_spec, f_spec)

# reference range start, stop step from one of the audio data
ref_range = {"time": (0, t_spec[-1], 0.1)}
extents = [
    # spectrogram subplot locations
    (0, 0.25, 0, 0.35),
    (0.25, 0.5, 0, 0.35),
    (0.5, 0.75, 0, 0.35),
    (0.75, 1, 0, 0.35),
    # behavior vid subplot locations
    (0, 0.25, 0.35, 1),
    (0.25, 0.5, 0.35, 1),
    (0.5, 0.75, 0.35, 1),
    (0.75, 1, 0.35, 1),
]

# %%
spec_names = list()
beh_names = list()
for loc in location_order:
    spec_names.append(f"spec-{loc}")
    beh_names.append(f"beh-{loc}")

ndw = fpl.NDWidget(
    ref_ranges=ref_range,
    extents=extents,
    names=[*spec_names, *beh_names],
    size=(1500, 800),
    controller_ids = None,#[[0, 0, 0, 0], [1, 2, 3, 4]]
)

cursor = fpl.Cursor()

def spec_tooltip_format(pick_info):
    col, row = pick_info["index"]



    return f"freq: {f_spec[row]}s"


for loc in location_order:
    spec_loc, t_spec, f_spec = specs[loc]
    spec_name = f"spec-{loc}"

    ng = ndw[spec_name].add_nd_timeseries(
        spec_loc,
        ("l", "time", "d"),
        ("l", "time", "d"),
        display_window=5.0,
        index_mappings={"time": t_spec},
        graphic=fpl.ImageGraphic,
        name=spec_name,
        x_range_mode="view-range",
        graphic_kwargs = {"metadata": {"f_spec": f_spec}}
    )
    ng.graphic.cmap = "viridis"
    ng.graphic.tooltip_format = spec_tooltip_format
    subplot = ndw.figure[spec_name]
    subplot.axes.y.tick_format = lambda v, min_v, max_v: f"{f_spec[max(min(round(v), 0), f_spec.size - 1)]} Hz"
    subplot.camera.maintain_aspect = False
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

    cursor.add_subplot(subplot)

    beh_name = f"beh-{loc}"
    vid, fps = movies[loc]
    ng = ndw[beh_name].add_nd_image(
        vid,
        dims=("time", "m", "n", "rgb"),
        spatial_dims=("m", "n", "rgb"),
        rgb_dim=("rgb"),
        index_mappings={"time": lambda timepoint: round(timepoint * fps)},
        compute_histogram=False,
        name=beh_name,
    )

ndw.show()

for subplot in ndw.figure:
    subplot.toolbar = False
    if "beh" in subplot.name:
        subplot.axes.visible = False
        subplot.camera.zoom = 1.25

fpl.loop.run()
