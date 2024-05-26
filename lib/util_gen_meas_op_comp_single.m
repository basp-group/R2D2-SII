function [A, At, G, W, nWimag, Wprecond] = util_gen_meas_op_comp_single(dataFilename, N, param_nufft, param_wproj, param_weight, param_precond)
% Build the measurement operator for a given uv-coverage at pre-defined
% frequencies.
%
% Parameters
% ----------
% dataFilename: function handle
%     Filenames of the data files to load ``u``, ``v`` and ``w`` coordinates and
%     ``nW`` the weights involved in natural weighting.
% N : int[1, 2]
%     Image dimension.
% param_nufft : struct
%     Structure to configure NUFFT.
% param_wproj : struct
%     Structure to configure w-projection
% param_weight : struct
%     Structure to configure data weighting


% Returns
% -------
% A : function handle
%     Function to compute the rescaled 2D Fourier transform involved
%     in the emasurement operator.
% At : function handle
%     Function to compute the adjoint of ``A``.
% G :  complex[:]
%     trimmed-down interpolation matrix.
% W :  bool[:]
%     selection vector.
% nWimag :  double[:]
%     Imaging weights (uniform/briggs)
% Wprecond :  double[:]
%     Preconditioning weights (uniform/briggs)

%% get vars
% load vars
load(dataFilename, 'u', 'v', 'w', 'nW');

% u v  are in units of the wavelength and will be normalised between [-pi,pi] for the NUFFT
u = double(u(:)) * pi / double(param_wproj.halfSpatialBandwidth);
v = -double(v(:)) * pi / double(param_wproj.halfSpatialBandwidth);
w = -double(w(:)); % !!  -1 to w coordinate
nW = double(nW(:));
nmeas = numel(u);

%% nufft operators
% compute dimensions
J = [param_nufft.Ky, param_nufft.Kx]; %  Dim. of the interpolation kernel
K = N .* [param_nufft.oy, param_nufft.ox]; % Dim. of the Fourier domain
nshift = N ./ 2; % Phase-shift Fourier space (expressed in number of samples in the Fourier domain).

% get sparse interpolation matrix `G` & nufft grid-correction `scale`
[G, scale] = createG([v, u], J, N, K, nshift, param_nufft.ktype);

% get Fourier operators
A = @(x) so_fft2(x, K, scale);
At = @(x) so_fft2_adj(x, N, K, scale);

%% inject weights in `G`
nWimag = 1; % init
% data weights for imaging
flag_gen_weights = (~(param_weight.flag_data_weighting))|| (~(param_weight.weight_load) && (param_weight.flag_data_weighting));

if param_weight.flag_data_weighting && param_weight.weight_load
    try load(dataFilename, 'nWimag');
        nWimag = double(nWimag(:));
    catch % update flag ..
        warning("imaging weights not found .. ")
        flag_gen_weights = true;
    end
end

if flag_gen_weights
    fprintf("\ngenerating imaging weights .. ")
    nWimag = util_gen_imaging_weights(u, v, nW, N, param_weight);
end

% inject weights in G
try G = (nWimag .* nW) .* G;
catch
    G = sparse(1:nmeas, 1:nmeas, (nWimag .* nW), nmeas, nmeas) * G;
end
clear nW;

%% apply w-correction via w-projection
if param_wproj.measop_flag_wproj && ~isempty(w) && nnz(w)
    param_nufft.J = J;
    param_nufft.K = K;
    param_nufft.N = N;
    G = wprojection_nufft_mat(G, w, param_nufft, param_wproj);
end

%% selection vector of non-empty cols in `G`
W = (abs(sum(G, 1).') > 0);

%% update `G` by removing empty cols
G = G(:, W);

%% compute uniform weights (sampling density) for  preconditioning
if exist('param_precond', 'var')
    Wprecond = util_gen_imaging_weights(u, v, [], N, param_precond).^2;
else
    Wprecond = [];
end

end
