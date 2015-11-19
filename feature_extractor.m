
% experiment 1 and 3 are guilty. Rest are not.
experiments = [1, 2, 3, 4, 5];
% stims[probe, target, irrelevant]
stims = [1, 2, 3];
% channels to get rid off: (50, 51, 65, 67)
channels_skip = [50, 51, 65, 66, 67];
% skipping 5 channels
numChannels = 67-5;
% add the mean feature per channel
featuresPerChannel = 276+1;
featureVectorSize = numChannels*featuresPerChannel;
% decided on the numImages based on the count which I ran the program once
numImages = 742;
count = 0;
lie_count=0;
featureMatrix = zeros(numImages, featureVectorSize);
combinedMeanMatrix = zeros(numImages, numChannels);
additionalSeeFeature = zeros(numImages,1);
ys = zeros(numImages,1);

for exp=1:numel(experiments)
    for stim=1:numel(stims)
        % whether guilty
        y = 0;
        if (exp == 1 || exp ==3) && stim==1
            y=1;
        end
%         feature indicating whether the subject saw the item in the image
%         before
        did_see = 0;
        if stim==2
            did_see = 1;
        end
%       number of images shown for particular stimulus in a given
%       experiment
        num_trials = 30;
        if stim==3
            num_trials = 90;
        end
        for trial=1:num_trials
           add_0 = '';
           if trial<10
               add_0 = '0';
           end
           file_location = strcat('/afs/ir.stanford.edu/users/b/a/bakis/Desktop/EEG/Stim_',num2str(stim));
           if exp ==1
               file_name = strcat('/data_Stim_',num2str(stim),'_trial0', add_0, num2str(trial),'.mat');
           else
               file_name = strcat('/data_Stim_',num2str(stim),'_trial0', add_0, num2str(trial),'_0',num2str(exp),'.mat');
           end
           file_path = strcat(file_location,file_name);
%            Check whether the file exists

           if exist(file_path, 'file') == 2
               count = count+1;
               file = load(file_path);
               imageDataMatrix = file.F;
               % get rid of channels 50, 51, 65, 66, 67
%              63 channels
               newImageDataMatrix = imageDataMatrix([1:49,52:64],:);
%                calculate mean value of all channels
               meansMatrix = mean(newImageDataMatrix, 2);
%                combine the channel values with means (each row has 276
%                values from each channel and 277th is the mean of the
%                first 276)
               imgEEGMatrix = horzcat(newImageDataMatrix,meansMatrix);
%                eegFeatureVector = reshape(imgEEGMatrix,1,[]);
               eegFeatureVector= [];
               for ch=1:size(imgEEGMatrix,1)
                 vec_row = imgEEGMatrix(ch,:);
                 eegFeatureVector = [eegFeatureVector vec_row];
               end
               featureMatrix(count,:) = eegFeatureVector;
               combinedMeanMatrix(count,:) = meansMatrix;
               additionalSeeFeature(count) = did_see;
%                add Y
               ys(count) = y;
               if y ==1
                    lie_count = lie_count+1;
               end
           end   
        end
    end
end

X = horzcat(featureMatrix,additionalSeeFeature);
y = ys;
save('feature_extractor.mat','X','y','combinedMeanMatrix')
hdf5write('eeg_X.h5', '/extractor', X);
hdf5write('eeg_y.h5', '/extractor', y);
hdf5write('eeg_means.h5', '/extractor', combinedMeanMatrix);