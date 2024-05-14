function [param_nufft, param_wproj, param_weight, param_precond] = util_set_param_operator(param_general, N, imPixelSize)

%% NUFFT
if ~isfield(param_general, 'nufft_oversampling')
    param_nufft.ox = 2; %zero-padding
    param_nufft.oy = 2; %zero-padding
else
    param_nufft.ox = param_general.nufft_oversampling(2); %zero-padding
    param_nufft.oy = param_general.nufft_oversampling(2); %zero-padding
end

if ~isfield(param_general, 'nufft_kernelDim')
    param_nufft.Kx = 7; %kernel dim 1
    param_nufft.Ky = 7; %kernel dim 2
else
    param_nufft.Kx = param_general.nufft_kernelDim(2); %kernel dim 1
    param_nufft.Ky = param_general.nufft_kernelDim(1); %kernel dim 2
end

if ~isfield(param_general, 'nufft_ktype') || isempty(param_general.nufft_ktype)
    param_nufft.ktype = 'minmax:kb';
else
    param_nufft.ktype = param_general.nufft_ktype;
end

fprintf("\nINFO: NUFFT kernel: Kaiser Bessel: size %d x %d, oversampling along each dim.: x%d, x%d", ...
    param_nufft.Kx, param_nufft.Ky, param_nufft.ox, param_nufft.oy)

%% w-projection functionality
if ~isfield(param_general, 'measop_flag_wproj') || isempty(param_general.measop_flag_wproj)
    param_wproj.measop_flag_wproj = false;
else
    param_wproj.measop_flag_wproj = param_general.measop_flag_wproj;
end

% sparsification params
if ~isfield(param_general, 'CEnergyL2') || isempty(param_general.CEnergyL2)
    param_wproj.CEnergyL2 = 1;
else
    param_wproj.CEnergyL2 = param_general.CEnergyL2;
end

if ~isfield(param_general, 'GEnergyL2') || isempty(param_general.GEnergyL2)
    param_wproj.GEnergyL2 = 1;
else
    param_wproj.GEnergyL2 = param_general.GEnergyL2;
end

% FoV-related params
param_wproj.pixelSize = imPixelSize;
param_wproj.FoVx = sin(imPixelSize*N(2)*pi/180/3600);
param_wproj.FoVy = sin(imPixelSize*N(1)*pi/180/3600);
param_wproj.uGridSize = 1 / (param_nufft.ox * param_wproj.FoVx);
param_wproj.vGridSize = 1 / (param_nufft.oy * param_wproj.FoVy);
param_wproj.halfSpatialBandwidth = (180 / pi) * 3600 / (imPixelSize) / 2;

%% data-weighting functionality
if ~isfield(param_general, 'flag_data_weighting') || isempty(param_general.flag_data_weighting)
    param_weight.flag_data_weighting = false;
else
    param_weight.flag_data_weighting = param_general.flag_data_weighting;
end

if ~isfield(param_general, 'weight_load') || ~param_weight.flag_data_weighting || isempty(param_general.weight_load)
    param_weight.weight_load = false;
else
    param_weight.weight_load = param_general.weight_load;
end

if ~param_weight.flag_data_weighting
    param_weight.weight_type = 'none';

elseif ~isfield(param_general, 'weight_type') && param_weight.flag_data_weighting
    param_weight.weight_type = 'uniform';
else
    param_weight.weight_type = param_general.weight_type;
end

if strcmp(param_weight.weight_type, 'briggs')
    if ~isfield(param_general, 'weight_robustness') || isempty(param_general.weight_robustness)
        param_weight.weight_robustness = 0;
    else
        param_weight.weight_robustness = param_general.weight_robustness;
    end
end

if ~isfield(param_general, 'weight_gridsize') || isempty(param_general.weight_gridsize)
    param_weight.weight_gridsize = max(param_nufft.oy, param_nufft.ox);
else
    param_weight.weight_gridsize = param_general.weight_gridsize;
end


% preconditionning params
if ~isfield(param_general, 'precond_weight_type') || isempty(param_general.precond_weight_type)
    param_precond.weight_type = 'uniform';
else
    param_weight.weight_type = param_general.precond_weight_type;
end

if ~isfield(param_general, 'precond_weight_gridsize') || isempty(param_general.precond_weight_gridsize)
    param_precond.weight_gridsize = max(param_nufft.oy, param_nufft.ox);
else
    param_precond.weight_gridsize = param_general.precond_weight_gridsize;
end


end
