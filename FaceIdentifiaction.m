clc;
clear;
close;


str = ['s (',int2str(1),')'];
my_image = imread(['train\ (',int2str(2),').png']);
im = my_image(:,:,[1 1 1]);
imwrite(imresize(im,[227,227]), ['Identification_photos\',str,'\',int2str(1),'.png']);
im = imageDatastore('Identification_photos','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
[Train ,Test] = splitEachLabel(im,0.8,'randomized');
fc = fullyConnectedLayer(1);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl; 
learning_rate = 0.0001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',5,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Train, ly, opts);


%% Org Test
correct = 0;

img = imread(['test\ (',int2str(1),').png']);
im = img(:,:,[1 1 1]);
img = imresize(im,[227 227]);
[predict,scores] = classify(newnet,img);
name = ['s (',int2str(1),')'];
if predict == name
    correct = 1;
end

% Accuracy = ((correct)/n)*100;
fprintf('The accuracy is: %f %% \n',correct*100);
