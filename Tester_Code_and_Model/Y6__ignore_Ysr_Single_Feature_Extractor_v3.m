 function [Retu_Features,Retu_Image, Orignal_String, Categorical_String] = Y6__ignore_Ysr_Single_Feature_Extractor_v3(image_path,image_text,Res50,layerRes50,GoogNet,layergoog)

% image_path=[fullfile(pwd,'!یوسف گیالنی سینیٹ الیکشن لڑ سکیں گے یا نہیں__$$2^^__757.png')];
% image_text='!یوسف گیالنی سینیٹ الیکشن لڑ سکیں گے یا نہیں__$$2^^__757';
% Res50 = resnet50;
% layerRes50 = 'activation_40_relu';
% 
% GoogNet = googlenet;
% layergoog='inception_4d-output';

No_Of_Records=1;
MaxCharSize_of_String=0;
StringSizesArray=zeros(1,No_Of_Records);
StringSpacesArray=zeros(1,No_Of_Records);
MaxSizeIndex=0;
MaxWidth=0;
MaxHeight=0;

BLI_rows=150;
BLI_cols=448;
BLI=uint8(zeros(BLI_rows,BLI_cols,3));

 Yasser_All_Rows=[];
 Yasser_All_Cols=[];
 Ysr_RowReSizeFlag=0;
 Ysr_ColReSizeFlag=0;
 CounterRowResize=1;
 CounterColResize=1;
 ReSizingFactor=0.9;
% keyboard
   title_text=image_text;
               Lu_img=imread(image_path);
            
               img_height=size(Lu_img,1);
               img_width=size(Lu_img,2);
                
                %////////////////////// Blank image Embedder////////////////////////////////////////////
                Original_image=Lu_img;
                yRow=img_height;
                xCol=img_width;
                Yasser_All_Rows=[Yasser_All_Rows;yRow];
                Yasser_All_Cols=[Yasser_All_Cols;xCol];
                while yRow > BLI_rows && CounterRowResize <4
                     Original_image=imresize(Original_image,[ floor(round(yRow*ReSizingFactor)) xCol]);
                     yRow=size(Original_image,1);
                     Ysr_RowReSizeFlag=1;
                    CounterRowResize=CounterRowResize+1;
                end

                while xCol > BLI_cols && CounterColResize <4
                     Original_image=imresize(Original_image,[ yRow floor(round(xCol*ReSizingFactor))]);
                     xCol=size(Original_image,2);
                     Ysr_ColReSizeFlag=1;
                     CounterColResize=CounterColResize+1;
                end
                  
                 if yRow > BLI_rows
                     yRow=BLI_rows;
                     Original_image=imresize(Original_image,[BLI_rows BLI_cols]);
%                      yRow;
                 end
                 if xCol > BLI_cols
                     xCol=BLI_cols;
                     Original_image=imresize(Original_image,[BLI_rows BLI_cols]);
% % %                      xCol
                 end
%                  subplot(1,2,1);
%                  imshow(Original_image);
                 if size(Original_image,3) > 1
                        BLI(1:yRow,1:xCol,:)=Original_image(1:yRow,1:xCol,:);
                 else
                        Original_image=cat(3,Original_image,Original_image,Original_image);
                        BLI(1:yRow,1:xCol,:)=Original_image;
                 end
% % % % % % % % % 
% % % % % % % % %                  subplot(1,2,2);
% % % % % % % % %                  imshow(BLI);
% % % % % % % % %                  title([title_text  ' --- ' ]);
% % % % % % % % %                  pause(0.1);
% % % % % % % %                  
             
% % % Res50 = resnet50;
% % % layerRes50 = 'activation_40_relu';
% % % 
% % % GoogNet = googlenet;
% % % layergoog='inception_4d-output';
% keyboard
             ResNet50featuresTrain=YNormalizeConcate_ResNet50_Feature_Extractor_From_Image( BLI,Res50,layerRes50);
             GoogfeaturesTrain =YNormalizeConcate_GoogleNet_Feature_Extractor_From_Image( BLI,GoogNet,layergoog); 
     
LSTM_LengthOfEach_Feature=128;
FeaturesLength=4753*128;

t1=ResNet50featuresTrain;
 t2=GoogfeaturesTrain;
 t_combination=[t1 t2];
 
% minWidthLabel=64;
minWidthLabel=128;                  %% it is used for target label setting but it also requires
                                    %% that features should be in form  4753 X 128 = 608384  (as number of features)
OriginalminWidthLabel=minWidthLabel;
maxWidthLabel=0;
NoOfFeaturesInLSTMinPut=floor(FeaturesLength/LSTM_LengthOfEach_Feature);  %% 319488/64 --> 4992
NoOfFeaturesInLSTMinPutUpper=ceil(FeaturesLength/LSTM_LengthOfEach_Feature);
% Diff_Upper_Lower=(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPutUpper)-(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPut)-2;
Diff_Upper_Lower=(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPutUpper)- size(t_combination,2);
% RemainderLength=mod(FeaturesLength,LSTM_LengthOfEach_Feature)

ZerosToBeAdded=[];
if Diff_Upper_Lower==0
    New_NoOfFeaturesInLSTMinPut=NoOfFeaturesInLSTMinPut;
else
    ZerosToBeAdded=zeros(1,Diff_Upper_Lower);
    New_NoOfFeaturesInLSTMinPut=NoOfFeaturesInLSTMinPutUpper;
end

% for pkp=1:size(ResNet50featuresTrain,2)
% % for pkp=1:38
    t1=ResNet50featuresTrain;
    t2=GoogfeaturesTrain;
    if size(ZerosToBeAdded,2)>=2
      t_combination=[t1 t2 ZerosToBeAdded];
    else
      t_combination=[t1 t2];
    end
    
    t1_reshape=reshape(t_combination(1:end),[New_NoOfFeaturesInLSTMinPut LSTM_LengthOfEach_Feature]);
    XT_Given_Image_Features={t1_reshape};
     

     a=unicode2native(string(image_text), 'UTF-8');
    % a(a==32)=65;
    % Yasser_Space_Addition=12;
    Yasser_Space_Addition=LSTM_LengthOfEach_Feature;
    if size(a,2)>=1
        SizeOfA=size(a,2);

        SizeDiff=Yasser_Space_Addition-SizeOfA;
        for kLu=1:SizeDiff
            a=[32 a];
        end
    end
     
     YasserCategoricalString = categorical(a);
     
     if size(image_text,2)< minWidthLabel
         minWidthLabel=size(image_text,2);
%          disp(['Minimum Width Label Found at index: ']);
     end
     if size(image_text,2)> maxWidthLabel
         maxWidthLabel=size(image_text,2);
%          disp(['Maximum Width Label Found at index: ']);
     end
     if size(image_text,2)> minWidthLabel
          temp=image_text;
          image_text=temp(1:minWidthLabel);   % Cutting Extra Labels (Jougaar Need to improve))
     end

Retu_Image=BLI;
Orignal_String=image_text;   
Categorical_String=YasserCategoricalString;
Retu_Features=XT_Given_Image_Features;
 end
             

             
             %%
%Local Funtions


% % function RetuNorCoFeaturesResNet50=YNormalizeConcate_ResNet50_Feature_Extractor_From_Image(ii,ResNet50,layerResnet50)
% %       %%
% %       % Part Half-A of image
% %       Half_A=ii(:,1:224);
% %       CurrentImage2=Y_Function_CLE_Normalize_ligatures_v7(Half_A,224,224);
% %       if size(CurrentImage2,3) > 1
% %          Temp_i=CurrentImage2;
% %       else
% %          Temp_i=cat(3,CurrentImage2,CurrentImage2,CurrentImage2);
% %       end
% %      RetuNorCoFeaturesResNet50_Half_A = activations(ResNet50,Temp_i,layerResnet50,'OutputAs','rows');
% %       %%
% %       % Part Half-B of image
% %       Half_B=ii(:,225:448);
% %       CurrentImage2=Y_Function_CLE_Normalize_ligatures_v7(Half_B,224,224);
% %       if size(CurrentImage2,3) > 1
% %          Temp_i=CurrentImage2;
% %       else
% %          Temp_i=cat(3,CurrentImage2,CurrentImage2,CurrentImage2);
% %       end
% %      RetuNorCoFeaturesResNet50_Half_B = activations(ResNet50,Temp_i,layerResnet50,'OutputAs','rows');
% %      
% %      RetuNorCoFeaturesResNet50 =[RetuNorCoFeaturesResNet50_Half_A RetuNorCoFeaturesResNet50_Half_B];
% % end

% function RetuNorCoFeaturesGoogleNet=YNormalizeConcate_GoogleNet_Feature_Extractor_From_Image(ii,GoogleNet,layerGoogleNet)
%       %%
%       % Part Half-A of image
%       Half_A=ii(:,1:224);
%       CurrentImage2=Y_Function_CLE_Normalize_ligatures_v7(Half_A,224,224);
%       if size(CurrentImage2,3) > 1
%          Temp_i=CurrentImage2;
%       else
%          Temp_i=cat(3,CurrentImage2,CurrentImage2,CurrentImage2);
%       end
%      RetuNorCoFeaturesGoogleNet_Half_A = activations(GoogleNet,Temp_i,layerGoogleNet,'OutputAs','rows');
%       %%
%       % Part Half-A of image
%       Half_B=ii(:,225:448);
%       CurrentImage2=Y_Function_CLE_Normalize_ligatures_v7(Half_B,224,224);
%       if size(CurrentImage2,3) > 1
%          Temp_i=CurrentImage2;
%       else
%          Temp_i=cat(3,CurrentImage2,CurrentImage2,CurrentImage2);
%       end
%      RetuNorCoFeaturesGoogleNet_Half_B = activations(GoogleNet,Temp_i,layerGoogleNet,'OutputAs','rows');
%      RetuNorCoFeaturesGoogleNet =[RetuNorCoFeaturesGoogleNet_Half_A  RetuNorCoFeaturesGoogleNet_Half_B];
% end
