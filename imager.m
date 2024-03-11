function imager(pathData, imDimx, imDimy, param_general,varargin)
%------------------------------------------------------------------
if ~isempty(varargin)
      param_groundtruth = varargin{1};
else, param_groundtruth =[];
end
%------------------------------------------------------------------
%% compute resources
if isfield(param_general,'nCpus') && ~isempty(param_general.nCpus)
    nCpuAvail=maxNumCompThreads;
    nCpuRequested = maxNumCompThreads(param_general.nCpus);
    fprintf("\nINFO: Available CPUs: %d.\nINFO: Requested CPUs: %d", nCpuAvail , maxNumCompThreads)
end
%------------------------------------------------------------------
%% setting paths
% main dir
projectDir = param_general.projectDir;
fprintf('\nINFO: Main project dir. is: %s', projectDir);
% matlab lib codes
addpath([projectDir, filesep, 'lib', filesep, 'lib_utils', filesep]);
addpath([projectDir, filesep, 'lib', filesep, 'RI-measurement-operator', filesep, 'nufft']);
addpath([projectDir, filesep, 'lib', filesep, 'RI-measurement-operator', filesep, 'irt', filesep, 'utilities']);
addpath([projectDir, filesep, 'lib', filesep, 'RI-measurement-operator', filesep, 'lib', filesep, 'utils']);
addpath([projectDir, filesep, 'lib', filesep, 'RI-measurement-operator', filesep, 'lib', filesep, 'operators']);
addpath([projectDir, filesep, 'lib', filesep, 'RI-measurement-operator', filesep, 'lib', filesep, 'ddes_utils']);
% results path
resultPath = param_general.resultPath;
srcName = param_general.srcName;
if ~isfolder(resultPath), mkdir(resultPath)
end
srcResultPath  = fullfile(resultPath,srcName);
if ~isfolder(srcResultPath), mkdir (srcResultPath);
end
fprintf('\nINFO: Results will be saved in: %s. ',srcResultPath)

%------------------------------------------------------------------%
%% setting general parameters
% general flags
if ~isfield(param_general, 'flagDataWeighting'),
    param_general.flagDataWeighting = false;
end
flagDataWeighting = param_general.flagDataWeighting;
%
if ~isfield(param_general, 'flagSaveAllOutputs'),
    param_general.flagSaveAllOutputs = false;
end
flagSaveAllOutputs = param_general.flagSaveAllOutputs;
%
if ~isfield(param_general, 'verbose'),
    param_general.verbose = true;
end
verbose  = param_general.verbose;
% super-resolution factor: the ratio between the given max projection baseline and the desired one
if ~isfield(param_general, 'nufftSuperresolution') && ((~isfield(param_general, 'imPixelSize') || isempty(param_general.imPixelSize)))
    param_general.nufftSuperresolution = 1.0;
end
nufftSuperresolution =  param_general.nufftSuperresolution;
%
if ~isfield (param_general,'imPixelSize')
    param_general.imPixelSize = [];
end
imPixelSize = param_general.imPixelSize;

%------------------------------------------------------------------

%% DATA
dataloaded = load(pathData, 'y', 'nW');
% apply natural weighting
DATA = double(dataloaded.y(:)) .*  double(dataloaded.nW(:)) ;
clear dataloaded;
% read imaging weights & apply them to the data
if flagDataWeighting
    try load(pathData,'nWimag');
        if ~isempty(nWimag)
            DATA = DATA .* double((nWimag(:)));
            clear nWimag;
            if verbose
                fprintf('\nINFO: Imaging weights applied to data.')
            end
        else
            flagDataWeighting = false;
            warning('imaging weights not found.');
        end
    catch, flagDataWeighting = false;
        warning('imaging weights not found.');
    end
end

%------------------------------------------------------------------%
%% Measurements & operators
ticmeas  = tic;
% Set pixel size
if isempty(imPixelSize)
    maxProjBaseline = double( load(pathData, 'maxProjBaseline').maxProjBaseline );
    spatialBandwidth = 2 * maxProjBaseline;
    imPixelSize = (180 / pi) * 3600 / (nufftSuperresolution * spatialBandwidth);
    fprintf('\nINFO: Pixel-size: %g arcsec, corresponding to super-resolution factor: %.2f  ',...
        imPixelSize, nufftSuperresolution);
else, fprintf('\nINFO: user-specified pixelsize: %g arcsec. ', imPixelSize)
end
% Set parameters releated to operators
[param_nufft, param_wproj] = util_set_param_operator(imDimx, imDimy, imPixelSize);
% Generate measurement operator and its adjoint
[A, At, G, W] = util_gen_meas_op(pathData, imDimx, imDimy, ...
    flagDataWeighting, param_nufft, param_wproj);
clear param_nufft  param_wproj;
tocmeas= toc(ticmeas);
if verbose
   fprintf('\nINFO: Time to build the measurement operator: %f sec ',tocmeas);
end
%------------------------------------------------------------------%
%% operators
MeasOp = @(x) forward_operator(x, G, W, A); % measurement op.
adjointMeasOp = @(y) adjoint_operator(y, G, W, At); % adjoint of the measurement op.
%------------------------------------------------------------------%
%% PSF
PSF = adjointMeasOp(MeasOp(full( sparse(floor(imDimy./2) + 1,floor(imDimx./2) + 1,1,imDimy,imDimx))));
max_PSF = max(PSF,[],'all');
if verbose, fprintf('\nINFO: PSF peak value: %g. ', max_PSF);
end
psfFile = fullfile(srcResultPath, 'psf.fits');
fitswrite(PSF,psfFile); % for info only


%------------------------------------------------------------------%
%------------------------------------------------------------------%
fprintf("\n____________________________________________________")
%% R2D2 imaging
%
dnnSeries = param_general.dnnSeries;
if strcmp(dnnSeries,"R3D3")
    fprintf("\nCompute PSF over twice the FoV for data fidelity layers in R2D2Net ..")
    ticpsf2  = tic;
    r3d3_oversampling  = 2;
    imDimx_2FoV = r3d3_oversampling.*imDimx;
    imDimy_2FoV = r3d3_oversampling.*imDimy;
    % Set parameters releated to operators
    [param_nufft_tmp, param_wproj_tmp] = util_set_param_operator(imDimx_2FoV, imDimy_2FoV, imPixelSize);
    % Generate measurement operator and its adjoint
    [A_tmp, At_tmp, G_tmp, W_tmp] = util_gen_meas_op(pathData, imDimx_2FoV , imDimy_2FoV, ...
        flagDataWeighting, param_nufft_tmp, param_wproj_tmp);
    clear param_nufft_tmp param_wproj_tmp;
    % operators
    MeasOp_tmp = @(x) forward_operator(x, G_tmp, W_tmp, A_tmp); % measurement op.
    adjointMeasOp_tmp = @(y) adjoint_operator(y, G_tmp, W_tmp, At_tmp); % adjoint of the measurement op.
    PSF_2FoV = adjointMeasOp_tmp(MeasOp_tmp(full( sparse(floor(imDimy_2FoV./2) + 1,floor(imDimx_2FoV./2) + 1,1,imDimy_2FoV,imDimx_2FoV))));
    PSF_2FoV = PSF_2FoV ./ max(PSF_2FoV,[],'all');
    clear A_tmp  At_tmp  G_tmp  W_tmp adjointMeasOp_tmp MeasOp_tmp;
    psf2FoVFile = fullfile(srcResultPath, 'PSF_2FoV.fits');
    fitswrite(PSF_2FoV,psf2FoVFile)
    tocpsf2= toc(ticpsf2);
    if verbose
       fprintf('\nINFO: Time to compute the PSF needed in R2D2Net: %f sec ',tocpsf2);
    end
    fprintf("\n____________________________________________________")
end

% R2D2 DNNs
nLayers = param_general.nLayers;
nIterations = param_general.nIterations; % number of DNNs in the series
ckptPath = param_general.ckptPath; % DNN checkpoint directory
dnnGPU = param_general.dnnGPU;
tic
dnn_series = cell(nIterations,1);
fprintf("\nLoading DNNs in matlab .. \n")
for itr  = 1:nIterations
    dnn_series{itr} = cell(nLayers,1);
    switch dnnSeries
        case "R2D2"
            end2endDnn = 'UNet' ;
	        dnnONNXFilefun = @(itr,iUnetBlock) fullfile(ckptPath,[dnnSeries,'_',end2endDnn,'_N',num2str(itr),'.onnx']);
        case "R3D3"
            end2endDnn = 'R2D2Net';
            fprefix = [dnnSeries,'_',end2endDnn,'_',num2str(nLayers),'Layers_N'];
            dnnONNXFilefun = @(itr,iUnetBlock) fullfile(ckptPath, [fprefix, num2str(itr)],[fprefix ,num2str(itr),'_L',num2str(iUnetBlock),'.onnx']);
    end
    for iUnetBlock = 1:nLayers
         dnn_series{itr}{iUnetBlock} = importONNXNetwork(dnnONNXFilefun(itr,iUnetBlock),'OutputLayerType','regression','GenerateCustomLayers',false);
    end
    fprintf("\n%s: %s DNN number %d loaded.",dnnSeries,end2endDnn, itr)
end
toc
fprintf("\n____________________________________________________")
fprintf("\n**R2D2 algorithm------------------------------------")


% Dirty image (i.e. normalized back-projected data)
ticComputeDirty = tic;
DirtyImNormalized = adjointMeasOp(DATA)./max_PSF;
ticComputeDirty = toc(ticComputeDirty);
fprintf('\nINFO: Time to compute the dirty image (normalized back-projected data): %f sec ',ticComputeDirty')
dirtyFile = fullfile(srcResultPath, 'dirty_image.fits');
fitswrite(DirtyImNormalized, dirtyFile);

% init time vars
totTimeModel = 0;
totTimeData = ticComputeDirty; % dirty image timing included.

% init
if dnnGPU
    fprintf("\nINFO: GPU DEVICE available.")
    % init gpu device & DNN input array
    DeviceGpu = gpuDevice;
    DNN_RESIDUAL_MODEL_3D =  zeros(imDimx,imDimy,2,'gpuArray');
    if strcmp(dnnSeries,"R3D3"),  PSF_2FoV_fft2  = fft2(gpuArray(PSF_2FoV));
    end
    wait(DeviceGpu)
else
    DNN_RESIDUAL_MODEL_3D =  zeros(imDimx,imDimy,2);
    if strcmp(dnnSeries,"R3D3"),  PSF_2FoV_fft2  = fft2(PSF_2FoV);
    end
end

%& algo iterations %&
for itr  = 1:nIterations
    ticUpdateModel = tic;
    %% update model
    fprintf('\nItr %d ------------------------', itr)
    fprintf('\nDNN inference & model update ..')
    % init input
    if itr == 1
        if dnnGPU
            DNN_RESIDUAL_MODEL_3D(:,:,1) =  gpuArray(DirtyImNormalized);
        else
            DNN_RESIDUAL_MODEL_3D(:,:,1) =  DirtyImNormalized;
        end
    end
    % image update
    switch dnnSeries
           case "R2D2"
               DNN_RESIDUAL_MODEL_3D = unet_inference(DNN_RESIDUAL_MODEL_3D, dnn_series{itr}{1}, itr);
           case "R3D3"
               DNN_RESIDUAL_MODEL_3D = r2d2net_inference(DNN_RESIDUAL_MODEL_3D, dnn_series{itr}, itr, PSF_2FoV_fft2);
    end

    % gather
    MODEL = double(gather(DNN_RESIDUAL_MODEL_3D(:,:,2))); % gather to cpu

    ticUpdateModel = toc(ticUpdateModel);
    if verbose,  fprintf('\nINFO: model update time:  %f sec ' , ticUpdateModel)
    end

    %% clear mem.
    dnn_series{itr} = []; % clear iteration-specific dnn

    %% residual dirty image
    fprintf('\nUpdate residual ..')
    ticUpdateResidual = tic;
    if dnnGPU
        DNN_RESIDUAL_MODEL_3D(:,:,1) = gpuArray(DirtyImNormalized - adjointMeasOp(MeasOp(MODEL))./ max_PSF) ;% compute residual
        wait(DeviceGpu) % for accurate timing
    else
        DNN_RESIDUAL_MODEL_3D(:,:,1) = DirtyImNormalized - adjointMeasOp(MeasOp(MODEL))./ max_PSF ;% compute residual
    end
    ticUpdateResidual = toc(ticUpdateResidual);

    %% update time & save images
    totTimeModel = totTimeModel + ticUpdateModel ;  % total time of model update step
    if itr < nIterations
        totTimeData  = totTimeData  + ticUpdateResidual ; % total time of data fidelity step
        if verbose, fprintf('\nINFO: residual update time: %f sec \n',ticUpdateResidual);
        end
    end

    % save current estimates (optional)
    if flagSaveAllOutputs
        RESIDUAL = double(gather(DNN_RESIDUAL_MODEL_3D(:,:,1))) ;
        if itr == 1
            tmp_R2D2ResidualFile = fullfile(srcResultPath, [srcName,'_',end2endDnn,'_residual_image.fits']);
            tmp_R2D2ModelFile = fullfile(srcResultPath, [srcName,'_',end2endDnn,'_model_image.fits']);
        else
            tmp_R2D2ResidualFile = fullfile(srcResultPath, [end2endDnn,'_tmp_itr',num2str(itr),'_residual_image.fits']);
            tmp_R2D2ModelFile = fullfile(srcResultPath, [end2endDnn,'_tmp_itr',num2str(itr),'_model_image.fits']);
        end
        fitswrite(double(MODEL),tmp_R2D2ModelFile);
        fitswrite(RESIDUAL,tmp_R2D2ResidualFile);
    end

end, clear dnn_series ;

% gather final images
RESIDUAL = double(gather(DNN_RESIDUAL_MODEL_3D(:,:,1))) ;
MODEL = double(gather(DNN_RESIDUAL_MODEL_3D(:,:,2))) ;
clear DNN_RESIDUAL_MODEL_3D;
fprintf("\n____________________________________________________")
fprintf("\n** Timings------------------------------------------")
fprintf('\n** Total imaging time: %f sec',  totTimeModel + totTimeData )
fprintf('\n** Total model update time: %f sec', totTimeModel )
fprintf('\n** Total data fidelity time: %f sec', totTimeData )
fprintf('\n** Average model update time per iteration  %f sec', totTimeModel/nIterations )
fprintf('\n** Average data fidelity time per iteration: %f sec', totTimeData/nIterations )
%-----------------------------------------------------------------
%% saving final results
R2D2ResidualFile = fullfile(srcResultPath, [srcName,'_',dnnSeries,'_residual_image.fits']);
R2D2ModelFile = fullfile(srcResultPath, [srcName,'_',dnnSeries,'_model_image.fits']);
fitswrite(RESIDUAL, R2D2ResidualFile );
fitswrite(MODEL, R2D2ModelFile);
% evaluation metrics
fprintf("\n____________________________________________________")
fprintf("\n** Evaluation metrics-------------------------------")
try
   if ~isempty(param_groundtruth)
     groundtruth = fitsread(param_groundtruth.groundtruthFile);
     snr= 20*log10(norm(groundtruth(:))./norm(groundtruth(:) - MODEL(:)));
     fprintf('\n** Image reconstruction SNR: %.4f dB',snr)
     if ~isempty(param_groundtruth.targetDynamicRange)
         DR = param_groundtruth.targetDynamicRange;
         parametric_log=@(x, a) max(x, [], 'all') .* log10(a .* x ./ max(x, [], 'all') + 1.) ./ log10(a);
         logsnr= 20*log10(norm(parametric_log(groundtruth(:), DR))./norm(parametric_log(groundtruth(:), DR) - parametric_log(MODEL(:), DR)));
             fprintf('\n** Image reconstruction logSNR at target dynamic range %.1f: %.4f dB', DR,logsnr)
     end
   end
end
fprintf('\n** Data fidelity: standard deviation of the residual dirty image : %f', std(RESIDUAL(:)))
sigma_residual = norm(RESIDUAL(:))./norm(DirtyImNormalized(:));
fprintf('\n** Data fidelity: sigma = ||residual|| / || dirty ||: %f',sigma_residual)
fprintf("\n____________________________________________________")
fprintf("\n____________________________________________________")


end
%########## Degridding  ################%
function y = forward_operator(x, G, W, A)
y = A(x);
y = G * y(W);
end
%########### Gridding  #################%
function x = adjoint_operator(y, G, W, At)
x = zeros(size(W,1), 1);
x(W) = ((y')* G)';
x = real(At(x));
end
%############# R3D3 image update #################%
function  out_ch1Residual_ch2Model= r2d2net_inference(out_ch1Residual_ch2Model, dnns, itr, PSF_2FoV_fft2)
nLayers = numel(dnns);
PSFDims = size(PSF_2FoV_fft2);
imDims = size(out_ch1Residual_ch2Model,[1 2]);
for iLayer = 1 : nLayers
    if itr ==1 && iLayer ==1
        nz_mean = mean(out_ch1Residual_ch2Model(:,:,1),'all'); % normalisation factor from dirty
        output  = predict(dnns{iLayer},out_ch1Residual_ch2Model./nz_mean) .* nz_mean;
        out_ch1Residual_ch2Model(:,:,2) = output;
    else
        nz_mean = mean(out_ch1Residual_ch2Model(:,:,2),'all'); % normalisation factor from model
        output  = predict(dnns{iLayer},out_ch1Residual_ch2Model./nz_mean) .* nz_mean;
        out_ch1Residual_ch2Model(:,:,2) = out_ch1Residual_ch2Model(:,:,2) + output;
    end
    output = real(fftshift(ifft2(fft2(output,PSFDims(1),PSFDims(2)).*PSF_2FoV_fft2)));
    out_ch1Residual_ch2Model(:,:,1) = out_ch1Residual_ch2Model(:,:,1) - output(1:imDims(1),1:imDims(2));
    clear  output;

end
% apply positivity
out_ch1Residual_ch2Model(:,:,2) = out_ch1Residual_ch2Model(:,:,2).*(out_ch1Residual_ch2Model(:,:,2)>0);
end
%############# R2D2 image update #################%
function  out_ch1Residual_ch2Model = unet_inference(out_ch1Residual_ch2Model, dnn, itr)
if itr == 1
    nz_mean = mean(out_ch1Residual_ch2Model(:,:,1),'all'); % normalisation factor from dirty
    out_ch1Residual_ch2Model(:,:,2)  = predict(dnn,out_ch1Residual_ch2Model./nz_mean) .* nz_mean;
else
    nz_mean = mean(out_ch1Residual_ch2Model(:,:,2),'all'); % normalisation factor from model
    out_ch1Residual_ch2Model(:,:,2)  = out_ch1Residual_ch2Model(:,:,2) + ...
        predict(dnn, flip(out_ch1Residual_ch2Model,3)./nz_mean) .* nz_mean;
end
% apply positivity
out_ch1Residual_ch2Model(:,:,2) = out_ch1Residual_ch2Model(:,:,2).*(out_ch1Residual_ch2Model(:,:,2)>0);
end
