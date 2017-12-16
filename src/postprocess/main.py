import ast
import argparse
import inspect
import json
import os
import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from enum import Enum, auto
import scipy.stats
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")
np.random.seed(sum(map(ord, "distributions")))


def to_json(data):
    return json.loads(data.replace(' ', ',')
                      .replace('(', '[')
                      .replace(')', ']'))


class Trace:
    def __init__(self, log_weight, list_values):
        """
        :param log_weight: float
        :param list_values: List of ndarray's of shape (hits, dim)
        """
        self._log_weight = log_weight
        self._list_values = list_values

    @staticmethod
    def parse(line):
        json_data = to_json(line)
        values = json_data[0]
        weight = json_data[1]
        list_values = []
        for k, v in values:
            if len(list_values) <= k:
                list_values.extend((k - len(list_values) + 1) * [[]])
            list_values[k].append(np.array(v))
        list_values = [np.asarray(v) for v in list_values]
        return Trace(log_weight=weight, list_values=list_values)

    @property
    def list_values(self):
        return self._list_values

    @property
    def log_weight(self):
        return self._log_weight


class DiscreteDistribution:
    def __init__(self, distribution):
        self._distribution = distribution

    @property
    def distribution(self):
        return self._distribution

    def domain(self):
        return self._distribution.keys()

    def __str__(self):
        return str(self._distribution)

class Empirical:
    def __init__(self, values, weights):
        if len(values) != len(weights):
            raise ValueError("The lists values and weights have to have the same size.")
        self._values = values
        self._weights = self._normalise_weights(weights)

    @staticmethod
    def _normalise_weights(weights):
        return weights / weights.sum()

    @property
    def n_points(self):
        return len(self._values)

    def mean(self):
        return np.average(self._values, weights=self._weights)

    def variance(self):
        mean = self.mean()
        return np.average(np.square(self._values-mean), weights=self._weights)

    def std(self):
        return np.sqrt(self.variance())

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        return self._weights

    def partial_distribution(self, n):
        return Empirical(self._values[:n], self._weights[:n])


class EmpiricalContinuous(Empirical):
    def __init__(self, values, weights):
        Empirical.__init__(self, values=values, weights=weights)

    def vals_weights_plot(self, eps=1.0e-8):
        big_enough = self._weights > eps
        return self.values[big_enough], self._normalise_weights(self.weights[big_enough])

    def is_discrete(self):
        return False

class EmpiricalDiscrete(DiscreteDistribution, Empirical):
    def __init__(self, values, weights, domain):
        Empirical.__init__(self, values=values, weights=weights)
        DiscreteDistribution.__init__(self, distribution=EmpiricalDiscrete._get_distr(values, weights, domain))

    @staticmethod
    def _get_distr(values, weights, domain):
        ret = {val: weights[values == val].sum() for val in set(values)}
        for elem in domain:
            if elem not in ret:
                ret[elem] = 0
        return ret

    def partial_distribution(self, n):
        e = Empirical.partial_distribution(self, n)
        return EmpiricalDiscrete(e.values, e.weights, self.domain())

    def is_discrete(self):
        return True

def homogeneous_arrays(lhs, rhs):
    """
    Returns arrays representing the distributions lhs and rhs in a common domain
    :param lhs: DiscreteDistribution
    :param rhs: DiscreteDistribution
    :return: np.ndarray
    """
    distr1 = lhs.distribution
    distr2 = rhs.distribution
    if distr1.keys() != distr2.keys():
        d1 = []
        d2 = []
        for key in set(distr1.keys()) | set(distr2.keys()):
            d1.append(distr1[key] if key in distr1 else 0)
            d2.append(distr2[key] if key in distr2 else 0)
        return np.array(d1, dtype=np.float32), \
               np.array(d2, dtype=np.float32)
    else:
        return np.array(list(distr1.values()), dtype=np.float32), \
               np.array(list(distr2.values()), dtype=np.float32)


def l2_distance(lhs, rhs):
    distr1, distr2 = homogeneous_arrays(lhs, rhs)
    return np.linalg.norm(distr1-distr2)


def kl_divergence(lhs, rhs, eps=1.0e-10):
    distr1, distr2 = homogeneous_arrays(lhs, rhs)
    return np.sum(np.multiply(distr1, np.log(np.divide(distr1, distr2 + eps) + eps)))

class Algorithm(Enum):
    SIS = auto()
    CSIS = auto()

    def file_name(self):
        if self == Algorithm.SIS:
            return "sis"
        elif self == Algorithm.CSIS:
            return "csis"

    def matplot_colour(self):
        if self == Algorithm.SIS:
            return "r"
        elif self == Algorithm.CSIS:
            return "y"


class Model:
    def __init__(self, folder, name):
        self._posteriors = {}
        self._ground_truth = []
        self._ids = []
        self._folder = folder
        self._name = name

    def load(self, root_folder="examples"):
        """
        Invariant: We always expect ids.data to have all the possible ids, this is,
        its size in lines is greater or equal to the one of truth.data
        """
        folder = "{}/{}".format(root_folder, self._folder)
        self.load_ids("{}/ids.data".format(folder))
        self.load_ground_truth("{}/truth.data".format(folder))
        for alg in Algorithm:
            self.load_posterior(alg, "{}/{}.data".format(folder, alg.file_name()))
        self._check_sizes()

    def _check_sizes(self):
        size = len(self._ids)
        ok = len(self._ground_truth) == size and all([len(distrs) == size for distrs in self._posteriors.values()])
        if not ok:
            print("The number of addresses for ids.data, truth.data and the empirical posteriors do not coincide!")
            print("If some ground truth value is not known or is a real value, please leave a blank line.")
            print("Ids size {}".format(len(self._ids)))
            print("Ground Truth size {}".format(len(self._ground_truth)))
            for alg, distrs in self._posteriors.items():
                print("{} size {}".format(alg.name, len(distrs)))
                diff = len(self._ids) - len(self._ground_truth)
                self._ground_truth.extend(diff*[None])

    def load_posterior(self, algorithm, file_name):
        """
        We assume that every trace has the same ids
        """
        try:
            with open(file_name, 'r') as f:
                traces = [Trace.parse(line) for line in f]
        except FileNotFoundError:
            return

        if not traces:
            raise RuntimeError("File " + file_name + " empty.")
        self._posteriors[algorithm] = []

        def log_to_weights(log_weights):
            return np.exp(log_weights - logsumexp(log_weights))

        weights = log_to_weights(np.array([tr.log_weight for tr in traces]))
        n = weights.size

        # List of predicts of length len(trace[0].values)
        # Each address is a ndarray of shape (hits, num_predicts, shape_predict)
        predicts = [np.empty(np.insert(vals.shape, 1, n), dtype=vals.dtype)
                    for vals in traces[0].list_values]
        for i, trace in enumerate(traces):
            for addr, vals in enumerate(trace.list_values):
                predicts[addr][:, i] = vals
        for predict in predicts:
            if issubclass(predict[0].dtype.type, np.integer):
                # we assume that for every address the possible values of the discrete distribution are the same
                domain = np.unique(predict)
                self._posteriors[algorithm].append(
                    [EmpiricalDiscrete(values=values, weights=weights, domain=domain) for values in predict])
            elif issubclass(predict[0].dtype.type, np.floating):
                distr_class = EmpiricalContinuous
                self._posteriors[algorithm].append(
                    [EmpiricalContinuous(values=values, weights=weights) for values in predict])
            else:
                self._posteriors[algorithm].append(None)
                continue

    def load_ids(self, file_name):
        with open(file_name, 'r') as f:
            self._ids = f.read().splitlines()

    def _load_distr(self, str):
        # God, this is dirty
        # We parse the name of the distribution and the arguments in a safe way
        try:
            ret = []
            body = ast.parse(str).body
            elements = body[0].value.elts
            # For now just for continuous. God, this is no good
            funcs_stats = [m[0] for m in inspect.getmembers(scipy.stats._continuous_distns, inspect.isclass)]
            for elem in elements:
                # We make sure that it is a function before retrieving it
                if "{}_gen".format(elem.func.id) not in funcs_stats:
                    return None
                fun = getattr(scipy.stats, elem.func.id)
                # ast.literal_eval is safe to evaluate
                ret.append(fun(*[ast.literal_eval(arg) for arg in elem.args]))
        except Exception:
            return None
        return ret

    def load_ground_truth(self, path):
        # Just for EmpiricalInt for now
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        json_data = to_json(line)
                        address = []
                        for hit in json_data:
                            address.append(DiscreteDistribution(distribution=dict(hit)))
                        self._ground_truth.append(address)
                    except json.decoder.JSONDecodeError:
                        distr = self._load_distr(line)
                        if distr is None:
                            print("Could not parse the line \n{}It should be a list separated with commas.".format(line))
                        self._ground_truth.append(distr)
                else:
                    self._ground_truth.append(None)

    def _plot_estimators(self, addr, output_folder, n_points=100):
        # Factorise coz itsx
        has_ground_truth = addr < len(self._ground_truth) and self._ground_truth[addr] is not None
        fig, axes = plt.subplots(2, len(self._ground_truth[addr]), squeeze=False)

        for j in range(axes.shape[1]):
            for alg, post in self._posteriors.items():
                distr = post[addr][j]
                n_samples = np.linspace(start=distr.n_points/n_points, stop=distr.n_points, num=n_points)
                n_samples = np.around(n_samples).astype(np.int)

                mean = [distr.partial_distribution(n).mean() for n in n_samples]
                print(min(distr.partial_distribution(n_samples[95])._weights[n_samples[94]:n_samples[95]]))
                print(np.argmax(distr.partial_distribution(n_samples[95])._weights[n_samples[94]:n_samples[95]]))
                print(126+95*(n_samples[95]-n_samples[94]))
                std = [distr.partial_distribution(n).std() for n in n_samples]
                axes[0, j].plot(n_samples, mean, color=alg.matplot_colour(), label=alg.name)
                axes[1, j].plot(n_samples, std, color=alg.matplot_colour(), label=alg.name)

            if has_ground_truth:
                ground_truth = self._ground_truth[addr][j]
                # It is the same for axes[0] and axes[1]
                limits = axes[0, j].get_xlim()
                n_samples = np.linspace(start=limits[0], stop=limits[1], num=n_points)
                n_samples = np.around(n_samples).astype(np.int)

                axes[0, j].plot(n_samples, np.full(n_samples.shape, ground_truth.mean()), color="g", label="Ground Truth")
                axes[1, j].plot(n_samples, np.full(n_samples.shape, ground_truth.std()), color="g", label="Ground Truth")
            axes[0, j].set_title("Mean")
            axes[1, j].set_title("Standard Deviation")
            axes[0, j].legend()
            axes[1, j].legend()

        plt.tight_layout()
        plt.savefig("{}/{}_estimators.png".format(output_folder, self._ids[addr].lower()), bbox_inches="tight")

    def _plot_continuous(self, addr, output_folder):
        has_ground_truth = addr < len(self._ground_truth) and self._ground_truth[addr] is not None
        fig, axes = plt.subplots(len(self._posteriors), len(self._ground_truth[addr]), squeeze=False)
        for j in range(axes.shape[1]):
            if has_ground_truth:
                ground_truth = self._ground_truth[addr][j]

            for i, (alg, post) in enumerate(self._posteriors.items()):
                distr = post[addr][j]
                vals, weights = distr.vals_weights_plot()
                sns.distplot(vals, kde=False, norm_hist=True, ax=axes[i][j], color=alg.matplot_colour(), hist_kws={'weights': weights}, label=alg.name)

                if has_ground_truth:
                    limits = axes[i, j].get_xlim()
                    x = np.linspace(limits[0], limits[1], 500)

                    # Hack to fit a distribution to the data
                    sample = np.random.choice(vals, 10000, p=weights)
                    params = ground_truth.dist.fit(sample)
                    pdf_fit = lambda val: ground_truth.dist.pdf(val, *params)

                    axes[i, j].plot(x, pdf_fit(x), "b", label="Best Fit")
                    axes[i, j].plot(x, ground_truth.pdf(x), "g", label="Ground Truth")
                    axes[i, j].legend()

            title = self._ids[addr]
            if len(post[addr]) > 1:
                title += " {}".format(i)
            axes[0, j].set_title(title)

        plt.tight_layout()
        plt.savefig("{}/{}.png".format(output_folder, self._ids[addr].lower()), bbox_inches="tight")

    def _accumulate(self, f, target, addr, n=None):
        distance = 0.0
        for empirical, ground_truth in zip(target[addr], self._ground_truth[addr]):
            distance += f(empirical.partial_distribution(n) if n is not None else empirical, ground_truth)
        return distance

    def _plot_funcs(self, addr, output_folder, n_points=100, log_scale=False):
        """
        We assume that for each algorithm all the distributions have the same number of points
        mean = self.mean()
        return np.average(np.square(self._values-mean), weights=self._weights)
        """
        funcs = [{"f": l2_distance,
                  "name": "l2_distance",
                  "label": "Sum of $\ell_2$ distances"
                  },
                 {"f": kl_divergence,
                  "name": "kl_divergence_1",
                  "label": "Sum of $D_{KL}(\widehat{\pi}, \pi)$"
                  },
                 {"f": lambda x, y: kl_divergence(y, x),
                  "name": "kl_divergence_2",
                  "label": "Sum of $D_{KL}(\pi, \widehat{\pi})$"
                  }]

        for f in funcs:
            fig, ax = plt.subplots()
            # First algorithm, first address, first hit
            distances = {}
            plot_fun = ax.loglog if log_scale else ax.plot
            for alg, posteriors in self._posteriors.items():
                n_particles = self._posteriors[alg][addr][0].n_points
                if log_scale:
                    n_samples = np.logspace(start=2, stop=np.log10(n_particles), num=n_points)
                else:
                    n_samples = np.linspace(start=n_particles/n_points, stop=n_particles, num=n_points)
                distances[alg] = np.array([self._accumulate(f["f"], posteriors, addr, n) for n in np.around(n_samples).astype(np.int)])
                plot_fun(n_samples, distances[alg], alg.matplot_colour(), label=alg.name)

            ax.yaxis.grid(b=True, which="both")
            ax.set_xlabel("Number of samples")
            ax.set_ylabel(f["label"])
            ax.legend(loc='upper right')
            ax.set_title(self._name)
            file_name = "{}/{}{}.png"
            fig.savefig(file_name.format(output_folder, f["name"], "_log" if log_scale else ""))

    def _plot_distibutions(self, addr, output_folder):
        distr_list = []
        if self._ground_truth[addr] is not None:
            distr_list.append((self._ground_truth, "Ground Truth"))
        if Algorithm.CSIS in self._posteriors:
            distr_list.append((self._posteriors[Algorithm.CSIS], "CSIS"))
        if Algorithm.SIS in self._posteriors:
            distr_list.append((self._posteriors[Algorithm.SIS], "SIS"))

        fig, axes = plt.subplots(len(distr_list))
        for idx, (ax, (distributions, plot_name)) in enumerate(zip(axes, distr_list)):
            keys = list(distributions[addr][0].distribution.keys())
            values = []
            ticks = []
            name = self._ids[addr]
            for i, distr in enumerate(distributions[addr]):
                ticks.append(name if len(distributions[addr]) == 1 else "{} {}".format(name, i))
                values.append(list(distr.distribution.values()))
            values = np.array(values).T
            im = ax.imshow(values, cmap="inferno", vmax=1.0, vmin=0.0)
            ax.yaxis.set_ticks(keys)
            ax.set_xticks(range(values.shape[1]))
            ax.set_title(plot_name)
            if idx != len(axes)-1:
                ax.set_xticklabels([])
            else:
                plt.setp(ax.get_xticklabels(), rotation=90)
                ax.set_xticklabels(ticks)

        plt.colorbar(im, ax=axes.ravel().tolist())

        file_name = "{}/discrete.png"
        fig.savefig(file_name.format(output_folder), bbox_inches="tight")

    def plot(self):
        output_folder = "img/{}".format(self._folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        a_posterior = next(iter(self._posteriors.values()))
        num_addr = len(a_posterior)
        for addr in range(num_addr):
            #if self._ground_truth[addr] is not None:
            if a_posterior[addr][0].is_discrete():
                self._plot_funcs(addr, output_folder=output_folder, log_scale=True)
                self._plot_funcs(addr, output_folder=output_folder, log_scale=False)
                if len(a_posterior[addr]) > 1:
                    self._plot_distibutions(addr, output_folder=output_folder)
            else:
                self._plot_continuous(addr, output_folder=output_folder)
                self._plot_estimators(addr, output_folder=output_folder)



def main():
    parser = argparse.ArgumentParser(description="Inference Compilation Plotting Utils")
    parser.add_argument("-f", "--folder_model", help="Folder from which to load the model", type=str)
    parser.add_argument("--name", help="Model name used in the graphs", default="Model", type=str)
    opt = pparser.parse_args()
    m = Model(folder=opt.folder_model, name=opt.name)
    m.load()
    m.plot()

if __name__ == "__main__":
    main()
