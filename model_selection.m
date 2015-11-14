clear
load('feature_extractor.mat')
cv = cvpartition(y,'holdout',0.3);
Xtrain = X(training(cv),:);
Ytrain = y(training(cv));
Xtest = X(test(cv),:);
Ytest = y(test(cv));

% running SVM
% SVMModel = fitcsvm(X, y);
% CVSVMModel = crossval(SVMModel);
% missClass = kfoldLoss(CVSVMModel);
% missClass
% missClass =
% 
%     0.0809



% running NB
% nb = fitNaiveBayes(X, y);
% p = nb.predict(X);
% cMat = confusionmat(y, p);
% cMat
% The within-class variance in each feature of TRAINING must be positive. The within-class variance in
% feature 18493 in class 1 are not positive.

