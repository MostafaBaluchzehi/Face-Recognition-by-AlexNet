clc;
clear;
close;

% n is the number of train images
n = 100;
%%
for i =1:n
    str = ['s (',int2str(i),')'];
    my_image = imread(['train\ (',int2str(i),').png']);
    im = my_image(:,:,[1 1 1]);
    imwrite(imresize(im,[227,227]), ['photos\',str,'\',int2str(i),'.png']);

end
%% Train
im = imageDatastore('photos','IncludeSubfolders',true,'LabelSource','foldernames');
im.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
[Train ,Test] = splitEachLabel(im,0.8,'randomized');
fc = fullyConnectedLayer(n);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl; 
learning_rate = 0.0001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',30,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Train, ly, opts);

%% Test
correct = 0;
for L =1:n
    img = imread(['test\ (',int2str(L),').png']);
    im = img(:,:,[1 1 1]);
    img = imresize(im,[227 227]);
    [predict,scores] = classify(newnet,img);
    name = ['s (',int2str(L),')'];
    if predict == name
        correct = correct + 1;
    end
end
Accuracy = ((correct)/n)*100;
fprintf('The accuracy is: %f %% \n',Accuracy);
