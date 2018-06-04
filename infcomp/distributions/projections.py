import torch
import torch.nn as nn
from infcomp.protocol import TableDistribution


class ProjectionType:
    def __init__(self, name, scalar=True):
        self.name = name
        self._scalar = scalar

    def pack(self, builder, mod, distr_name, value):
        getattr(mod, "{}Add{}".format(distr_name, self.name))(builder, value)

    def pack_value(self, builder, mod, distr_name, value):
        # Flatbuffer is horrible
        if self._scalar:
            return value.data.squeeze().item()
        else:
            # TODO(Lezcano) Chech if there's a better way of doing this
            vec_numpy = value.data.cpu().numpy().squeeze()
            data = vec_numpy.flatten().tolist()
            getattr(mod, "{}Start{}Vector".format(distr_name, self.name))(builder, len(data))
            for d in reversed(data):
                builder.PrependFloat64(d)
            vector_fbb = builder.EndVector(len(data))
            return vector_fbb


class DefaultProjection(nn.Module):
    def __init__(self, dim, input_dim):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            #nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            #nn.BatchNorm1d(input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            #nn.BatchNorm1d(input_dim//8),
            nn.ReLU(),
            nn.Linear(input_dim//8, dim)
        )

    def forward(self, x):
        return self._model(x)


class ScalarHyperparamType(ProjectionType):
    def __init__(self, name, distribution_fbb):
        super().__init__(name, scalar=True)
        self.value = getattr(distribution_fbb, name)()

    def pack_value(self, builder, mod, distr_name):
        return self.value


class VectorHyperparamType(ProjectionType):
    def __init__(self, name, distribution_fbb):
        super().__init__(name, scalar=False)
        raise NotImplementedError("value parsing not implemented. Refactoring needed to use unpack_vector in module prior_distributions")
        #self.value = value

    def pack_value(self, builder, mod, distr_name, value):
        return ProjectionType.pack_value(self, builder, mod, distr_name, self.value)


# TODO(Lezcano) This should probably be a metaclass
class DistributionProjection(ProjectionType, nn.Module):
    def __init__(self, name, input_dim, distribution_type, distribution_fbb):
        ProjectionType.__init__(self, name, scalar=False)
        nn.Module.__init__(self)
        self._distr = distribution_type(distribution_fbb=distribution_fbb, input_dim=input_dim)

    def forward(self, x):
        return self._distr(x)

    def pack_value(self, builder, mod, distr_name, value):
        distr_fbb, distr_type_fbb = self._distr.serialize(builder, value)
        TableDistribution.TableDistributionStart(builder)
        TableDistribution.TableDistributionAddDistribution(builder, distr_fbb)
        TableDistribution.TableDistributionAddDistributionType(builder, distr_type_fbb)
        return TableDistribution.TableDistributionEnd(builder)

    def pack(self, builder, mod, distr_name, value):
        super(DistributionProjection, self).pack(builder, mod, distr_name, value)

    def log_pdf(self, x, *distribution_params):
        return self._distr.log_pdf(x, *distribution_params)

    def cdf(self, x, *distribution_params):
        return self._distr.cdf(x, *distribution_params)


class ProjectionList(ProjectionType, nn.ModuleList):
    def __init__(self, name, projections):
        ProjectionType.__init__(self, name, scalar=True)
        nn.ModuleList.__init__(self, projections)

    def forward(self, x):
        return [proj(x) for proj in self]

    def pack_value(self, builder, mod, distr_name, value):
        values_fbb = [proj.pack_value(builder, mod, distr_name, v) for proj, v in zip(self, value)]
        getattr(mod, "{}Start{}Vector".format(distr_name, self.name))(builder, len(values_fbb))
        for value_fbb in reversed(values_fbb):
            builder.PrependUOffsetTRelative(value_fbb)
        return builder.EndVector(len(values_fbb))

    def log_pdf(self, x, distribution_params):
        return torch.cat([proj.log_pdf(x, *params) for proj, params in zip(self, distribution_params)], dim=1)

    def cdf(self, x, distribution_params):
        return torch.cat([proj.cdf(x, *params) for proj, params in zip(self, distribution_params)], dim=1)


class LogSimplexProjection(ProjectionType, DefaultProjection):
    def __init__(self, dim, name, input_dim, exponentiate=False):
        ProjectionType.__init__(self, name, scalar=False)
        DefaultProjection.__init__(self, dim=dim, input_dim=input_dim)
        self._exponentiate = exponentiate
        self._logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = super(LogSimplexProjection, self).forward(x)
        return self._logsoftmax(x)

    def pack_value(self, builder, mod, distr_name, value):
        if self._exponentiate:
            value = torch.exp(value)
        return ProjectionType.pack_value(self, builder, mod, distr_name, value)


class SimplexProjection(ProjectionType, DefaultProjection):
    def __init__(self, name, input_dim, dim):
        ProjectionType.__init__(self, name, scalar=False)
        DefaultProjection.__init__(self, dim=dim, input_dim=input_dim)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = super(SimplexProjection, self).forward(x)
        return self._softmax(x)


class RealProjection(ProjectionType, DefaultProjection):
    def __init__(self, name, input_dim):
        ProjectionType.__init__(self, name, scalar=True)
        DefaultProjection.__init__(self, dim=1, input_dim=input_dim)


class RealPositiveProjection(ProjectionType, DefaultProjection):
    def __init__(self, name, input_dim):
        ProjectionType.__init__(self, name, scalar=True)
        DefaultProjection.__init__(self, dim=1, input_dim=input_dim)
        self._relu = nn.Softplus()

    def forward(self, x):
        x = super(RealPositiveProjection, self).forward(x)
        return self._relu(x)


class RealInIntervalProjection(ProjectionType, DefaultProjection):
    # Creates a value in an interval [a, b]
    def __init__(self, name, input_dim, a, b):
        ProjectionType.__init__(self, name, scalar=True)
        DefaultProjection.__init__(self, dim=1, input_dim=input_dim)
        self._a = a
        self._b = b
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super(RealInIntervalProjection, self)(x)
        return self._sigmoid(x)*(self._b - self._a) + self._a
