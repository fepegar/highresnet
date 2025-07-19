import numpy as np

from torch.utils.data import Dataset


class GridSampler(Dataset):
    """
    Adapted from NiftyNet
    """

    def __init__(self, data, window_size, border):
        self.array = data
        self.locations = self.grid_spatial_coordinates(
            self.array,
            window_size,
            border,
        )

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
        window = self.array[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        window = window[np.newaxis, ...]  # add channels dimension
        sample = dict(
            image=window,
            location=location,
        )
        return sample

    @staticmethod
    def _enumerate_step_points(starting, ending, win_size, step_size):
        starting = max(int(starting), 0)
        ending = max(int(ending), 0)
        win_size = max(int(win_size), 1)
        step_size = max(int(step_size), 1)
        if starting > ending:
            starting, ending = ending, starting
        sampling_point_set = []
        while (starting + win_size) <= ending:
            sampling_point_set.append(starting)
            starting = starting + step_size
        additional_last_point = ending - win_size
        sampling_point_set.append(max(additional_last_point, 0))
        sampling_point_set = np.unique(sampling_point_set).flatten()
        if len(sampling_point_set) == 2:
            sampling_point_set = np.append(
                sampling_point_set, np.round(np.mean(sampling_point_set))
            )
        _, uniq_idx = np.unique(sampling_point_set, return_index=True)
        return sampling_point_set[np.sort(uniq_idx)]

    @staticmethod
    def grid_spatial_coordinates(array, window_shape, border):
        shape = array.shape
        num_dims = len(shape)
        grid_size = [
            max(win_size - 2 * border, 0)
            for (win_size, border) in zip(window_shape, border)
        ]
        steps_along_each_dim = [
            GridSampler._enumerate_step_points(
                starting=0,
                ending=shape[i],
                win_size=window_shape[i],
                step_size=grid_size[i],
            )
            for i in range(num_dims)
        ]
        starting_coords = np.asanyarray(np.meshgrid(*steps_along_each_dim))
        starting_coords = starting_coords.reshape((num_dims, -1)).T
        n_locations = starting_coords.shape[0]
        # prepare the output coordinates matrix
        spatial_coords = np.zeros((n_locations, num_dims * 2), dtype=np.int32)
        spatial_coords[:, :num_dims] = starting_coords
        for idx in range(num_dims):
            spatial_coords[:, num_dims + idx] = (
                starting_coords[:, idx] + window_shape[idx]
            )
        max_coordinates = np.max(spatial_coords, axis=0)[num_dims:]
        assert np.all(max_coordinates <= shape[:num_dims]), (
            "window size greater than the spatial coordinates {} : {}".format(
                max_coordinates, shape
            )
        )
        return spatial_coords


class GridAggregator:
    """
    Adapted from NiftyNet
    """

    def __init__(self, data, window_border):
        self.window_border = window_border
        self.output_array = np.full(
            data.shape,
            fill_value=0,
            dtype=np.uint16,
        )

    @staticmethod
    def crop_batch(windows, location, border=None):
        if not border:
            return windows, location
        location = location.astype(np.int)
        batch_shape = windows.shape
        spatial_shape = batch_shape[2:]  # ignore batch and channels dim
        num_dimensions = 3
        for idx in range(num_dimensions):
            location[:, idx] = location[:, idx] + border[idx]
            location[:, idx + 3] = location[:, idx + 3] - border[idx]
        if np.any(location < 0):
            return windows, location

        cropped_shape = np.max(location[:, 3:6] - location[:, 0:3], axis=0)
        diff = spatial_shape - cropped_shape
        left = np.floor(diff / 2).astype(np.int)
        i_ini, j_ini, k_ini = left
        i_fin, j_fin, k_fin = left + cropped_shape
        if np.any(left < 0):
            raise ValueError
        batch = windows[
            :,  # batch dimension
            :,  # channels dimension
            i_ini:i_fin,
            j_ini:j_fin,
            k_ini:k_fin,
        ]
        return batch, location

    def add_batch(self, windows, locations):
        windows = windows.cpu()
        location_init = np.copy(locations)
        init_ones = np.ones_like(windows)
        windows, _ = self.crop_batch(
            windows,
            location_init,
            self.window_border,
        )
        location_init = np.copy(locations)
        _, locations = self.crop_batch(
            init_ones,
            location_init,
            self.window_border,
        )
        for window, location in zip(windows, locations):
            window = window.squeeze()
            i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            self.output_array[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = window
