clear
load('feature_extractor.mat')

preNN = y;
inputMatrix = X(:,1:17174);
targetVector = y;

net = patternnet(6);
net = train(net, inputMatrix', targetVector');
y2 = net(inputMatrix');
per = perform(net, targetVector', y2);
