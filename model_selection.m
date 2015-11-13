clear
load('feature_extractor.mat')
cv = cvpartition(y,'holdout',0.3);
Xtrain = X(training(cv),:);
Ytrain = y(training(cv));
Xtest = X(test(cv),:);
Ytest = y(test(cv));
