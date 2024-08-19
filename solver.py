import numpy as np


class Solver(object):
    def __init__(self):
        self._filter_size = 3

    @staticmethod
    def sliding_window_view(x, shape, subok=False, writeable=False):
        x = np.array(x, copy=False, subok=subok)

        shape = np.array(shape, np.intp)

        step = np.ones(len(x.shape), np.intp)

        o = (np.array(x.shape) - shape) // step + 1  # output shape

        strides = x.strides
        view_strides = strides * step

        view_shape = np.concatenate((o, shape), axis=0)
        view_strides = np.concatenate((view_strides, strides), axis=0)
        view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)

        return view

    def erode(self, img, kernel_size: int = 3) -> np.ndarray:
        img = img.astype(bool)
        st_element = np.ones((kernel_size, kernel_size), dtype=bool)

        padded = np.pad(img, st_element.shape[0] // 2)

        windows = self.sliding_window_view(padded, shape=(kernel_size, kernel_size))

        eroded = np.all(windows | ~st_element, axis=(-1, -2))

        return eroded

    def median_filter(self, img, kernel_size: int) -> np.ndarray:
        padded = np.pad(img, kernel_size // 2)
        return np.median(self.sliding_window_view(padded, shape=(kernel_size, kernel_size)), axis=(-1, -2))

    @staticmethod
    def hough_line(img, angle_step=1) -> tuple:
        thetas = np.deg2rad(np.arange(-90, 90, angle_step))
        width, height = img.shape

        diag_len = int(round(np.sqrt(width * width + height * height)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)

        accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
        y_idxs, x_idxs = np.nonzero(img)

        xcosthetas = np.dot(x_idxs.reshape((-1, 1)), cos_t.reshape((1, -1)))
        ysinthetas = np.dot(y_idxs.reshape((-1, 1)), sin_t.reshape((1, -1)))
        rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
        rhosmat = rhosmat.astype(np.int16)
        for i in range(num_thetas):
            rhos, counts = np.unique(rhosmat[:, i], return_counts=True)
            accumulator[rhos, i] = counts
        return accumulator, thetas, rhos

    @staticmethod
    def get_peaks(accumulator, thetas) -> tuple:
        rhos = np.linspace(-707, 707, 1414)
        maxi = accumulator[np.argmax(accumulator, axis=0), np.arange(thetas.shape[0])]
        max_split_idx = np.zeros(6, dtype=int)
        for idx, array in enumerate(np.array_split(maxi, 6)):
            max_split_idx[idx] = idx * 30 * 4 + np.argmax(array)
        max_split_idx = max_split_idx[(maxi[max_split_idx]).argsort()[-3:]]
        max_count = maxi[max_split_idx][(maxi[max_split_idx]).argsort()[-3:]]

        thetas_0 = thetas[max_split_idx[0]]
        thetas_1 = thetas[max_split_idx[1]]
        thetas_2 = thetas[max_split_idx[2]]

        rhos_0 = float(
            rhos[np.where(accumulator[:, max_split_idx[0]] == max_count[0])[0]][0]
        )
        rhos_1 = float(
            rhos[np.where(accumulator[:, max_split_idx[1]] == max_count[1])[0]][0]
        )
        rhos_2 = float(
            rhos[np.where(accumulator[:, max_split_idx[2]] == max_count[2])[0]][0]
        )
        return thetas_0, thetas_1, thetas_2, rhos_0, rhos_1, rhos_2

    @staticmethod
    def get_line_intersection(rho_1, theta_1, rho_2, theta_2) -> np.ndarray:
        cos_sin_thetas = np.array([
            [np.cos(theta_1), np.sin(theta_1)],
            [np.cos(theta_2), np.sin(theta_2)]
        ])

        rhos = np.array([[rho_1], [rho_2]])
        line = np.linalg.solve(cos_sin_thetas, rhos).flatten()

        return line

    def solve(self, img: np.ndarray) -> np.ndarray:
        black_ratio = img[img == 0].shape[0] / (500 * 500)

        # 9
        if black_ratio < 0.18:
            img = self.erode(img, 7)
        # 8
        elif 0.18 <= black_ratio < 0.28:
            img = self.erode(img, 3)
        # 5, 6, 7
        elif 0.28 <= black_ratio < 0.76:
            img = self.erode(img, 3)
        # 2, 3, 4
        elif 0.76 <= black_ratio < 0.86:
            img = self.erode(img, 2)
        else:
            img = np.where(img > 0, 255, 0)

        accumulator, thetas, rhos = self.hough_line(img, 0.25)
        thetas_0, thetas_1, thetas_2, rhos_0, rhos_1, rhos_2 = self.get_peaks(
            accumulator, thetas
        )

        line_0 = self.get_line_intersection(rhos_0, thetas_0, rhos_1, thetas_1)
        line_1 = self.get_line_intersection(rhos_1, thetas_1, rhos_2, thetas_2)
        line_2 = self.get_line_intersection(rhos_0, thetas_0, rhos_2, thetas_2)
        return np.array([line_0, line_1, line_2])


