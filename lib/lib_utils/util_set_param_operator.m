function [param_nufft, param_wproj] = util_set_param_operator(imDimx, imDimy, imPixelSize)
    
% NUFFT: default vals
param_nufft.ox = 2; %zero-padding
param_nufft.oy = 2;  %zero-padding
param_nufft.Kx = 7;   %kernel dim 1
param_nufft.Ky = 7;   %kernel dim 2

% FoV info for w-proj
param_wproj.measop_flag_wproj = false; % hard-coded for now
param_wproj.CEnergyL2 = 1;
param_wproj.GEnergyL2 = 1;
param_wproj.FoVx = sin(imPixelSize * imDimx * pi / 180 / 3600);
param_wproj.FoVy = sin(imPixelSize * imDimy * pi / 180 / 3600);
param_wproj.uGridSize = 1 / (param_nufft.ox * param_wproj.FoVx);
param_wproj.vGridSize = 1 / (param_nufft.oy * param_wproj.FoVy);
param_wproj.halfSpatialBandwidth = (180 / pi) * 3600 / (imPixelSize) / 2;

end