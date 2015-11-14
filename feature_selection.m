clear
load('feature_extractor.mat')
% X1 = X(:,17730:end);
% X1 = X;
% coeff = pca(X);
c = cvpartition(y,'k',10);
opts = statset('display','iter');
fun = @(XT,yT,Xt,yt)...
    (sum(~strcmp(yt,classify(Xt,XT,yT,'linear'))));

[fs,history] = sequentialfs(fun,X,y,'cv',c,'direction', 'backward', 'options', opts);

% http://stats.stackexchange.com/questions/53096/issues-with-sequential-feature-selection
% [fs1, history] = sequentialfs(@SVM_class_fun, X, y, 'cv', c);
% function err = SVM_class_fun(xTrain, xTest, yTrain, yTest)
%   model = svmtrain(xTrain, yTrain, 'Kernel_Function', 'rbf', 'boxconstraint', 10);
%   err = sum(svmclassify(model, xTest) ~= yTest);
% end
