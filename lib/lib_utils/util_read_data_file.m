function y = util_read_data_file(dataFilename, flag_data_weighting)
dataloaded = load(dataFilename, 'y', 'nW');

% apply natural weights
y = double(dataloaded.y(:)) .* double(dataloaded.nW(:));


end