
import mnist1d


def load_mnist1d():
    args = mnist1d.data.get_dataset_args()
    data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=True)
    return data['x'], data['y'], data['x_test'], data['y_test']