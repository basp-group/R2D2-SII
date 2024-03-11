function run_imager(json_filename)
% Read configuration parameters defined in an input ``.json`` file & Run imager
% Parameters
% ----------
% json_filename : string
%     Name of the .json configuration file.
% Returns
% -------
% None
%% Parsing json file
clc; close all;
fid = fopen(json_filename);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config = jsondecode(str);

%%  main input
main = cell2struct(struct2cell(config{1, 1}.main), fieldnames(config{1, 1}.main));
% set project directory
if ~isfield(main, 'projectDir') || isempty(main.projectDir)
    main.projectDir = fullfile(pwd, filesep);
end
% set results path 
if ~isfield(main, 'resultPath') || isempty(main.resultPath)
    main.resultPath = fullfile(main.dirProject,filesep, 'results');
end
% verbose
if ~isfield(main, 'verbose') || isempty(main.verbose)
    main.verbose = true;
end
% save all files
if ~isfield(main, 'flagSaveAllOutputs') || isempty(main.flagSaveAllOutputs)
    main.flagSaveAllOutputs = true;
end

%% measurement operator
param_measop = cell2struct(struct2cell(config{2, 1}.measop), fieldnames(config{2, 1}.measop));
%  critical imaging params related to pixel resolution
if (~isfield(param_measop,'imPixelSize') || isempty(param_measop.imPixelSize)) && ...
    (~isfield(param_measop,'nufftSuperresolution') || isempty(param_measop.nufftSuperresolution))
    warning("information on pixel resolution not provided,  will assume instrumental resolution. ")
    param_measop.nufftSuperresolution  = 1;
elseif (isfield(param_measop,'imPixelSize') && ~isempty(param_measop.imPixelSize)) && ...
    (isfield(param_measop,'nufftSuperresolution') && ~isempty(param_measop.nufftSuperresolution))
    fprintf("\nINFO: the provided pixelsize in arcsec will be considered for pixel resolution. ")
end

if  main.verbose
    disp("param measurement op.:")
    disp(param_measop)
end

%% algorithm
param_solver = cell2struct(struct2cell(config{3, 1}.r2d2), fieldnames(config{3, 1}.r2d2));
% check dnn folder
if ~isfolder(param_solver.ckptPath), error('CRITICAL: DNNs directory  not found.');
end
% check gpu device
if (~isfield(param_solver,'dnnGPU') || isempty(param_solver.dnnGPU))
   param_solver.dnnGPU = (gpuDeviceCount("available")>0);
end

if  main.verbose
    disp("param algorithm:")
    disp(param_solver)
end
%% full param list
param_general = cell2struct([ struct2cell(param_measop); struct2cell(param_solver)], ...
    [fieldnames(param_measop); fieldnames(param_solver)]);

%%  groundtruth file (optional)
param_groundtruth = [] ;
try param_groundtruth =  cell2struct(struct2cell(config{4, 1}.sim), fieldnames(config{4, 1}.sim));
    if isfield(param_groundtruth,'groundtruthFile') && ~isempty(param_groundtruth.groundtruthFile)
       if ~isfield(param_groundtruth,'targetDynamicRange')
           param_groundtruth.targetDynamicRange = [];
       end
    end
    % check file exists
    if  ~isempty(param_groundtruth.groundtruthFile) && ~exist(param_groundtruth.groundtruthFile,'file'),
        param_groundtruth = [];
        warning('groundtruth file not found.')
    end
end

%% parse some of the main params
param_general.projectDir = main.projectDir;
param_general.resultPath = main.resultPath;
param_general.srcName = main.srcName;
param_general.verbose = main.verbose;
param_general.flagSaveAllOutputs = main.flagSaveAllOutputs;
%% main function
fprintf("\n____________________________________________________")
% compulsory imaging settings
imDimx = main.imDimx;
imDimy = main.imDimy;
dataFile = main.dataFile;
% check data file
%if ~isfile(dataFile), error('CRITICAL: data file not found.');
%end
% run imager
imager(dataFile, imDimx, imDimy, param_general,param_groundtruth);
fprintf('\nTHE END. \n')
end
