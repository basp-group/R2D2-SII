function [A, At, G, W] = util_gen_meas_op(dataFilename, imDimx, imDimy, flag_data_weighting, param_nufft, param_wproj)
                                                                     % nDataSets, ddesfilename
    % Build the measurement operator for a given uv-coverage at pre-defined
    % frequencies.
    %
    % Parameters
    % ----------
    % dataFilename: function handle
    %     Filenames of the data files to load ``u``, ``v`` and ``w`` coordinates and
    %     ``nW`` the weights involved in natural weighting.
    % effChans2Image: cell
    %     Indices of the  physical (input) channels, combined to image the
    %     effective (ouput) channel.
    % nDataSets :int
    %     Number of datasets per physical (input) channel.
    % Nx : int
    %     Image dimension (x-axis).
    % Ny : int
    %     Image dimension (y-axis).
    % param_nufft : struct
    %     Structure to configure NUFFT.
    % param_wproj : struct
    %     Structure to configure w-projection.
    % param_precond : struct
    %     Structure to configure the preconditioning matrices.
    % ddesfilename: function handle
    %     Filenames of the DDE calibration kernels in the spatial Fourier domain
    %     from a pre-processing step to be incorporated in the measurement
    %     operator. Expected variable to be loaded ``DDEs``.
    %

    % Returns
    % -------
    % A : function handle
    %     Function to compute the rescaled 2D Fourier transform involved
    %     in the emasurement operator.
    % At : function handle
    %     Function to compute the adjoint of ``A``.
    % G : matrix of complex[:]
    %     matrix containing the trimmed-down interpolation kernels for each
    %     channel, and each data block within a channel.
    % W : matrix of double[:]
    %     matrix containing the selection vector for each channel, and
    %     data block within a channel.
    % nWimag : vector [:]
    %     vector containing the sqrt of imaging weights (uniform/briggs) for each channel, and
    %     data block within a channel.
    %%
    speed_of_light = 299792458;

    param_nufft.N = [imDimy, imDimx];
    param_nufft.Nn = [param_nufft.Ky, param_nufft.Kx];
    param_nufft.No = [param_nufft.oy * imDimy, param_nufft.ox * imDimx];
    param_nufft.Ns = [imDimy / 2, imDimx / 2];
    
    % load noise  and data weights
    if flag_data_weighting
        load(dataFilename, 'nW', 'nWimag');
        nWimag = (double(nWimag(:)));
        nW = (double(nW(:))) .* nWimag;
        clear nWimag;
    else
        load(dataFilename,  'nW');
        nW = double(nW(:));
    end
    
    % load uvw

    load(dataFilename, 'u','v','w');
    

    % u v  are in units of the wavelength and will be normalised between [-pi,pi] for the NUFFT
    u = double(u(:)) * pi / double(param_wproj.halfSpatialBandwidth);
    v = -double(v(:)) * pi / double(param_wproj.halfSpatialBandwidth);
    w = -double(w(:)); % !! add -1 to w coordinate


    % measurement operator initialization
    [A, At, G, W] = op_p_nufft_wproj_dde(param_nufft, [{v} {u}], {w}, {nW}, param_wproj);
    clear u v w nW;
    G = G{1};
    W = W{1};
    

   
end
