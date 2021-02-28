clc;
clear;
close;

% n is the number of train images
n = 63;

for i =1:n
    str = ['s (',int2str(i),')'];
    my_image = imread(['train\ (',int2str(i),').png']);
    im = my_image(:,:,[1 1 1]);
    imwrite(imresize(im,[227,227]), ['photos\',str,'\',int2str(i),'.png']);

end
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


%% Org Test
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
fprintf('The accuracy of the 2 Up Bic is: %f %% \n',Accuracy);
%% 2 Bic
% correct = 0;
% for L =1:n
%     img = imread(['test_bic2\ (',int2str(L),').png']);
%     dims = size(img);
%     dims = dims(1:2);
%     dims = round(dims/2);
%     img = imresize(img, dims);
%     im = img(:,:,[1 1 1]);
%     img = imresize(im,[227 227]);
%     [predict,scores] = classify(newnet,img);
%     name = ['s (',int2str(L),')'];
%     if predict == name
%         correct = correct + 1;
%     end
% end
% Accuracy = ((correct)/n)*100;
% fprintf('The accuracy of the 2 Up Bic is: %f %% \n',Accuracy);

%% 4 Bic
% correct = 0;
% for L =1:n
%     img = imread(['test_bic4\ (',int2str(L),').png']);
% %     dims = size(img);
% %     dims = dims(1:2);
% %     dims = round(dims/4);
% %     img = imresize(img, dims);
%     im = img(:,:,[1 1 1]);
%     img = imresize(im,[227 227]);
%     [predict,scores] = classify(newnet,img);
%     name = ['s (',int2str(L),')'];
%     if predict == name
%         correct = correct + 1;
%     end
% end
% Accuracy = ((correct)/n)*100;
% fprintf('The accuracy of the 4 Up Bic is: %f %% \n',Accuracy);

%% 2 MLP
% correct = 0;
% for L =1:n
%     img = imread(['test_mlp2\ (',int2str(L),').png']);
% %     dims = size(img);
% %     dims = dims(1:2);
% %     dims = round(dims/8);
% %     img = imresize(img, dims);
%     im = img(:,:,[1 1 1]);
%     img = imresize(im,[227 227]);
%     [predict,scores] = classify(newnet,img);
%     name = ['s (',int2str(L),')'];
%     if predict == name
%         correct = correct + 1;
%     end
% end
% Accuracy = ((correct)/n)*100;
% fprintf('The accuracy of the 2 MLP is: %f %% \n',Accuracy);

%% 4 MLP
% correct = 0;
% for L =1:n
%     img = imread(['test_mlp4\ (',int2str(L),').png']);
% %     dims = size(img);
% %     dims = dims(1:2);
% %     dims = round(dims/12);
% %     img = imresize(img, dims);
%     im = img(:,:,[1 1 1]);
%     img = imresize(im,[227 227]);
%     [predict,scores] = classify(newnet,img);
%     name = ['s (',int2str(L),')'];
%     if predict == name
%         correct = correct + 1;
%     end
% end
% Accuracy = ((correct)/n)*100;
% fprintf('The accuracy of the 4 MLP is: %f %% \n',Accuracy);

%% 20
% correct = 0;
% for L =1:n
%     img = imread(['test\ (',int2str(L),').png']);
%     dims = size(img);
%     dims = dims(1:2);
%     dims = round(dims/20);
%     img = imresize(img, dims);
%     im = img(:,:,[1 1 1]);
%     img = imresize(im,[227 227]);
%     [predict,scores] = classify(newnet,img);
%     name = ['s (',int2str(L),')'];
%     if predict == name
%         correct = correct + 1;
%     end
% end
% Accuracy = ((correct)/n)*100;
% fprintf('The accuracy of the 20 Low is: %f %% \n',Accuracy);

