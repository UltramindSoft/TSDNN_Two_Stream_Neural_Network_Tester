% clc
% clear all
% close all

tic
    Res50 = resnet50;
    layerRes50 = 'activation_40_relu';

    GoogNet = googlenet;
    layergoog='inception_4d-output';
    
PreTraind_Transfer_Models_Loading_Time=toc;
tic
    load('Sr_Model.mat');
BLSTM_Model_Loading_Time=toc;

image_path=[fullfile(pwd,'بیسٹ بائے کیش اینڈ کیری__$$11^^__1_100023.png')];
image_text='بیسٹ بائے کیش اینڈ کیری__$$11^^__1_100023';

% image_path=[fullfile(pwd,'لیڈیز__$$29^^__1_100062.png')];
% image_text='لیڈیز__$$29^^__1_100062';

% image_path=[fullfile(pwd,'مظفرآباد__$$7^^__3_100012.png')];
% image_text='مظفرآباد__$$7^^__3_100012';

% image_path=[fullfile(pwd,'منہاس__$$52^^__2_100131.png')];
% image_text='منہاس__$$52^^__2_100131';

% image_path=[fullfile(pwd,'نعمان فاروق__$$2^^__1_100002.png')];
% image_text='نعمان فاروق__$$2^^__1_100002';
% 
% image_path=[fullfile(pwd,'دستیاب ہے__$$53^^__2_100135.png')];
% image_text='دستیاب ہے__$$53^^__2_100135';


disp('Recognizing Urdu Text in an image Takes a little time... so b patient');
    tic
    % [Retu_Image, Orignal_String, Categorical_String] =Ysr_Single_image_Embedder_n_Feature_Extractor_v2(image_path,image_text);
    [XT_Given_Image_Features,Retu_Image, Orignal_String, Categorical_String]= ...
        Y6__ignore_Ysr_Single_Feature_Extractor_v3(image_path,image_text,Res50,layerRes50,GoogNet,layergoog);
    End_time_Feature_Builder=toc;

     tic
        YPred = classify(net,XT_Given_Image_Features, ...
                         'MiniBatchSize',1, ...
                         'SequenceLength','longest');
    End_time_Classifying_the_Urdu_Patch=toc;

    
        reverse_bytes1_Prediction_seq={};
        for klucid=1:size(YPred,1)
        % for klucid=1:size(YPred,1)
            tt=YPred{klucid,1};
            tt2=cellstr(tt);
            tt3=cell2table(tt2');
            ych=char(tt3.Var1');
            uru=native2unicode(str2num(ych),'utf-8');
            reverse_bytes1_Prediction_seq{klucid} = uru;
        %     reverse_bytes1{klucid} = native2unicode(uint16(YPred{klucid,1}), 'UTF-8')
        end

Converted_Categorical_String= reverse_bytes1_Prediction_seq;
PredictedString=Converted_Categorical_String{1,1};
Final_Urdu_Predicted_String=PredictedString';

PreTraind_Transfer_Models_Loading_Time
BLSTM_Model_Loading_Time
disp(['End_time_Feature_Builder :' num2str(End_time_Feature_Builder) ' Sec   ,  End_time_Classifying_the_Urdu_Patch :' num2str(End_time_Classifying_the_Urdu_Patch) ' Sec']);
Final_Urdu_Predicted_String

% whos all