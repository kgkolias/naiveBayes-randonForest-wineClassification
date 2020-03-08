wine_data = readtable('wineScaled.csv'); %Load the data

%Save the column names for later use
elements = {'Alcohol'; 'Malic acid'; 'Ash'; 'Alcalinity of ash'; 'Magnesium'; 'Total phenols'; 'Flavanoids'; 'Nonflavanoid phenols'; 'Proanthocyanins'; 'Color intensitys'; 'Hue'; 'OD280/OD315 of diluted wines'; 'Proline'};
    
[m n] = size(wine_data); %saving the dimensions of my dataset
fprintf('The wine dataset consists of %d Rows and %d Columns.\n', m, n')

Y = wine_data.Type; %The target 'Type'
X = wine_data(:,[2:end]); %The Predictors

%Implement the Minimum redundancy selection algorithm to find the relevance
%of each attribute with the other ones and visualise it in a plot
[idx, scores] = fscmrmr(X,Y); % using fscmrmr
bar(scores(idx));
xlabel('Predictor rank');
ylabel('Predictor importance score');
xticklabels(strrep(X.Properties.VariableNames(idx),'_','\_'));
xtickangle(60);

%Dropping the columns that are highly correlated with the 'Type' column,
%based on the barplot and the heatmap created (poster)

wine_data.Flavanoids = [];
wine_data.AlcalinityOfAsh = [];
wine_data.OD280_OD315OfDilutedWines = [];
wine_data.MalicAcid = [];
wine_data.Hue = [];
wine_data.NonflavanoidPhenols = [];
wine_data.Proanthocyanins = []
wine_data.TotalPhenols = [];
wine_data.Proline = [];


Y = wine_data.Type;
X = wine_data(:,[2:end]);

%Print the new plot
[idx, scores] = fscmrmr(X,Y);
bar(scores(idx));
xlabel('Predictor rank');
ylabel('Predictor importance score');
xticklabels(strrep(X.Properties.VariableNames(idx),'_','\_'));
xtickangle(60);

rng(1) %for reproducibility purposes
random_index = randperm(m); %randomised index
p=0.7; %the proportion of the training data

%Splitting the dataset wine_data in 2 sets: training_set and testing_set
training_set = wine_data(random_index(1:round(p*m)),:); %Keep the 70% of the data for training purposes
testing_set = wine_data(random_index(round(p*m)+1:end),:); %Keep the 30% of the data for testing purposes

%The 70% of the dataset that is going to be used for Training purposes
X_training = training_set(:,[2:end]);
Y_training = training_set(:,1);

%The 30% of the dataset that is going to be used for Testing purposes
X_testing = testing_set(:,[2:end]);
Y_testing = testing_set(:,1);

X1 = table2array(X_training);
Y1 = table2array(Y_training);

X2 = table2array(X_testing);
Y2 = table2array(Y_testing);
%Dividing the Training set in training and test validation, with k number
%of Folds
k=10; %The number of Folds
partitions = cvpartition(Y1, 'KFold', k, 'Stratify', true);

%% STAGE 1 - Naive Bayes with Default Values of Hyperparameters

prior = [0.5 0.2 0.3];

confusionTotalDefaultNB = zeros(3,3)
tic
for i = 1:k
    trainIndexesDefaultNB = partitions.training(i);
    testIndexesDefaultNB = partitions.test(i);
    defaultModelNB = fitcnb(X1(trainIndexesDefaultNB,:),Y1(trainIndexesDefaultNB,:), 'Prior', prior);
    [labels,probabilities] = predict(defaultModelNB,X1(testIndexesDefaultNB,:));
    cross_labels{i} = Y1(testIndexesDefaultNB);
    cross_scores{i} = probabilities(:,2);
    confusionDefaultNB = confusionmat(Y1(testIndexesDefaultNB),labels); %confusion matrix fed with expected and predicted values
    fprintf('this is the confusion matrix number %.2f\n',i)
    confusionDefaultNB
    accDefaultNB(i) = accuracy_calc(confusionDefaultNB) %calculating the accuracy based on the default NB confusion matrix
    confusionTotalDefaultNB = confusionTotalDefaultNB + confusionDefaultNB % calculating the confusion matrix for all the folds
    
end
toc

matDefaultNB = confusionchart(confusionTotalDefaultNB);
fprintf('the mean accuracy is %.2f\n',mean(accDefaultNB))
fprintf('the max accuracy is %.2f\n',max(accDefaultNB))
partitions
size(Y)
defaultModelNB.Prior
%% STAGE 2 - NAIVE BAYES with hyperparameter tuning

%Declaration of the Hyperparameters for training
Distribution = ["Kernel" "Normal"]; % Distribution hyperparameter which is going to take 2 values: kernel and normal
distribution_length = length(Distribution);

Width = [0.1 0.3 0.5 0.7 0.9 1.1 1.3]; %declaration of the bandwidth hyperparameter
width_length = length(Width);

prior = [0.5 0.2 0.3];
prior_length = length(prior);

%Declaration of the Naive Bayes Hyperparameters
hyperparKernelNB = [];
hyperparNormalNB = [];
confusionTotalKernel = zeros(3,3);
confusionTotalNormal = zeros(3,3);
counterKernel = 0
counterNormal = 0
tic
for p = 1:prior_length
    for y = 1:distribution_length %iteration for different distribution types
        if Distribution(y) == "Kernel" %If we have a Kernel Distribution
            for i = 1:k %iterate inside the folds
                for j = 1:width_length %iterate based on different bandwidth values
                    trainIndexesKernel = partitions.training(i); %saving the training part from our partitions
                    testIndexesKernel = partitions.test(i); %saving the test part from our partitions
                    %Using fitcnb method with the appropriate
                    %Hyperparameters for Kernel
                    modelNB = fitcnb(X1(trainIndexesKernel,:),Y1(trainIndexesKernel,:),'Width', Width(j),'DistributionNames',[{'kernel'},{'kernel'},{'kernel'},{'kernel'}]);
                    [labels,probabilities] = predict(modelNB,X1(testIndexesKernel,:)); %storing the labels and the probabilities
                    cross_labels_Kernel{i} = Y1(testIndexesKernel); %saving the cross validation labels for Kernel Distribution
                    cross_scores_Kernel{i} = probabilities(:,2); %saving the cross validation scores for Kernel Distribution
                    confusionKernel = confusionmat(Y1(testIndexesKernel),labels); %confusion matrix with expected and predicted values
                    confusionTotalKernel = confusionTotalKernel + confusionKernel; %calculating the total confusion matrix for Kernel Distribution
                    accKernel(i) = accuracy_calc(confusionKernel);
                    hyperparKernelNB = [hyperparKernelNB; Width(j), accKernel(i)]; %calculating a table with every hyperparameter combination for Kernel
                    counterKernel = counterKernel + 1;
                end
            end
            accKernelTotal = accuracy_calc(confusionTotalKernel);
            partitions;
            matKernel = confusionchart(confusionTotalKernel);

            % Sorting the values of the hypertable based on the Accuracy in ascending order. The best accuracies are the ones in the latest positions.
            [~,idx] = sort(hyperparKernelNB(:,2));
            hyperparKernelNB = hyperparKernelNB(idx,:);
        else                                           %if the distribution is Normal
            for i=1:k
                trainIndexesNormal = partitions.training(i); %saving the training part from our partitions
                testIndexesNormal = partitions.test(i); %saving the test part from our partitions
                %Using fitcnb method with the appropriate Hyperparameters
                %for Normal
                modelNB = fitcnb(X1(trainIndexesNormal,:),Y1(trainIndexesNormal,:), 'Prior', prior); 
                [labels,probabilities] = predict(modelNB,X1(testIndexesNormal,:)); 
                cross_labels_Normal{i} = Y1(testIndexesNormal);
                cross_scores_Normal{i} = probabilities(:,2);
                confusionNormal = confusionmat(Y1(testIndexesNormal),labels); %confusion matrix fed with expected and predicted values
                fprintf('this is the confusion matrix number %.2f\n',i)
                confusionNormal
                accNormal(i) = accuracy_calc(confusionNormal);
                confusionTotalNormal = confusionTotalNormal + confusionNormal;
                hyperparNormalNB = [hyperparNormalNB; accNormal(i), prior(p)];
                counterNormal = counterNormal + 1;
            end
            accNormalTotal = accuracy_calc(confusionTotalNormal);
            accNormalTotal
            fprintf('The mean accuracy Normal is %.2f\n',mean(accNormal))
            fprintf('The max accuracy Normal is %.2f\n',max(accNormal))
            partitions;
            modelNB.Prior;

            confusionTotalNormal;
            matNormal = confusionchart(confusionTotalNormal);
            matNormal;

            [~,idx] = sort(hyperparNormalNB(:,1)); %Sorting hypertables values based on the Accuracy in ascending order. The best accuracies are the ones in the latest positions.
            hyperparNormalNB = hyperparNormalNB(idx,:);  %In this occasion we don't use another hyperparameter, so they take the default values (Normal)
        end
    end
end
toc

% Confusion Matrix for Naive Bayes with Normal Distribution
matNBNormal = confusionchart(confusionTotalNormal);
matNBNormal.title('Confusion Matrix for Naive Bayes with Normal Distribution')

%% STAGE 3 - NAIVE BAYES BEST HYPERPARAMETERS FOR NAIVE BAYES
%Taking the last row of the sorted Hyperparameters combinations based on the Maximum Accuracy

optimal_hyperparams_Normal_NB = hyperparNormalNB(end,:);
optimal_hyperparams_Kernel_NB = hyperparKernelNB(end,:);
%% Training the best Normal distributed Naive Bayes model and testing it to the testing set

optimal_modelNormalNB = fitcnb(X1,Y1,'Prior', prior); %fitting the training predictors and target
[labels,probabilities] = predict(optimal_modelNormalNB,X2); %using the testing set predictors for the model prediction 
cross_labels_optimal_Normal = Y2; %target in the testing set
cross_scores_optimal_Normal = probabilities(:,2); %scores based on the probabilities calculated
optimal_confusionNormalNB = confusionmat(Y2,labels); %calculating the confusion matrix of the best hyperparameters combination for Normal Distribution
optimal_accNormalNB = accuracy_calc(optimal_confusionNormalNB);
optimal_accNormalNB

%Classification Error for Naive Bayes classifier using the best combination
%of Hyperparameters
loss_Normal = loss(optimal_modelNormalNB,X2,Y2);
%% Training the best Kernel distributed Naive Bayes model and testing it to the testing set

optimal_modelKernelNB = fitcnb(X1,Y1, 'Width', optimal_hyperparams_Kernel_NB(1),'DistributionNames',[{'kernel'},{'kernel'},{'kernel'},{'kernel'}]);
[labels,probabilities] = predict(optimal_modelKernelNB,X2); %using the testing set predictors for the model prediction 
cross_labels_optimal_Kernel = Y2; %target in the testing set
cross_scores_optimal_Kernel = probabilities(:,2); %scores based on the probabilities calculated
optimal_confusionKernelNB = confusionmat(Y2,labels);
optimal_accKernelNB = accuracy_calc(optimal_confusionKernelNB);
optimal_accKernelNB

%% STAGE 4 - Plotting the ROC Curves for Naive Bayes -Normal, Kernel, Optimal-
%ROC Curve for Naive Bayes Normal Distribution
[FP,TP,T,AUC, OPTROCPT]= perfcurve(cross_labels_Normal,cross_scores_Normal,2);
fprintf('AUC = %f\n',AUC);
figure
plot(FP,TP)
hold on
plot((0:0.1:1), (0:0.1:1));
hold off
legend('Type 1', 'Type 2', 'Type 3');
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Naive Bayes Classifier with Normal Distribution')

%ROC Curve for Naive Bayes Kernel Distribution
[FP,TP,T,AUC, OPTROCPT]= perfcurve(cross_labels_Kernel,cross_scores_Kernel,2);
fprintf('AUC = %f\n',AUC);
figure
plot(FP,TP)
hold on
plot((0:0.1:1), (0:0.1:1));
hold off
legend('Type 1', 'Type 2', 'Type 3');
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Naive Bayes Classifier with Kernel Distribution')

%ROC Curve for Naive Bayes Best Model with Optimal Hyperparameters 
[FP,TP,T,AUC, OPTROCPT]= perfcurve(cross_labels_optimal_Normal,cross_scores_optimal_Normal, 2);
fprintf('AUC = %f\n',AUC);
figure
plot(FP,TP)
hold on
plot((0:0.1:1), (0:0.1:1));
hold off
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Naive Bayes Classifier with Optimal Hyperparameters')
%% STAGE 1 - RANDOM FOREST with Default Hyperparameters
%Declaration of Hyperparameters
numTrees = [10 20 40 80 160 320];
counterDefaultRF = 0;
confusionTotalDefaultRF = zeros(3,3);
tic
for i = 1:k
    for j = 1:length(numTrees)
        trainIndexesDefaultRF = partitions.training(i); %saving the training part from our partitions
        testIndexesDefaultRF = partitions.test(i); %saving the test part from our partitions
        %Using Treebagger for our model training
        modelDefaultRF = TreeBagger(numTrees(j), X1(trainIndexesDefaultRF,:), Y1(trainIndexesDefaultRF,:),'OOBPrediction', 'on')
        [labels,probabilities] = predict(modelDefaultRF,X1(testIndexesDefaultRF,:)); %storing the labels and the probabilities
        labels = str2double(labels);
        modelDefaultRF_error = oobError(modelDefaultRF, 'Mode', 'Ensemble'); %default model error
        sum_errors = oobError(modelDefaultRF);
        cross_labels{i} = Y1(testIndexesDefaultRF); %calculating the RF labels
        cross_scores{i} = probabilities(:,2); %calculating the RF scores
        confusionDefaultRF = confusionmat(Y1(testIndexesDefaultRF),labels);
        accDefaultRF = accuracy_calc(confusionDefaultRF);
        cross_labelsDefaultRF{i} = Y1(testIndexesDefaultRF);
        cross_scoresDefaultRF{i} = probabilities(:,2);
        confusionDefaultRF = confusionmat(Y1(testIndexesDefaultRF),labels); %confusion matrix fed with expected and predicted values
        
        counterDefaultRF = counterDefaultRF + 1
    end
    confusionTotalDefaultRF = confusionTotalDefaultRF + confusionDefaultRF
    accTotalDefaultRF = accuracy_calc(confusionTotalDefaultRF)
end
toc
%% %% STAGE 2 - RANDOM FOREST
%Random Forest Algorithm with Hyperparameters

%Declaration of Hyperparameters for Random Forest
rng(1) %for reproducibility purposes
numTrees = [10 20 40 80 160 320]; %declaration of different numbers of trees
minLeaf = [1:10]; %minimum  number of observations per leaf in the tree
numFeatures = [1:4]; %number of features that have been used
%Initialisations
counterRF = 0
confusionTotalRF = zeros(3,3);
accuracyRF = [];
hyperparRF = [];
tree_errors = [];
hyperTable = zeros(600, 4);

%Grid Search using the RF Hyperparameters using 10-fold cross validation
tic
for l = 1:k
    for y = 1:length(numTrees) 
        for i = 1:length(minLeaf)
            for j = 1:length(numFeatures)
                trainIndexesRF = partitions.training(l);
                testIndexesRF = partitions.test(l);
                modelRF = TreeBagger(numTrees(y),X1(trainIndexesRF,:), Y1(trainIndexesRF,:),'OOBPrediction', 'on', 'minLeafSize', minLeaf(i), ...
                    'NumPredictorsToSample', numFeatures(j));
                hyperparRF = [hyperparRF; numTrees(y), minLeaf(i), numFeatures(j)]; %storing the hypermeters combinations
                [labels,probabilities] = predict(modelRF,X1(testIndexesRF,:)); %calculating the labels and the probabilitites based on the model that has been trained
                labels = str2double(labels);
                modelRF_error = oobError(modelRF);
                tree_errors = [tree_errors; modelRF_error];
                confusionRF = confusionmat(Y1(testIndexesRF),labels); %calculating the confusion matrix based on the predicted and the expected values of the target
                accRF = accuracy_calc(confusionRF); % calculating the accuracy based on the confusion matrix
                accuracyRF = [accuracyRF; accRF]; %storing each accuracy for every hyperparameters combination
                cross_labelsRF{l} = Y1(testIndexesRF); %calculating the labels per each fold
                cross_scoresRF{l} = probabilities(:,2); %calculating the scores per each fold
                confusionRF = confusionmat(Y1(testIndexesRF),labels); %confusion matrix with expected and predicted values
                confusionTotalRF = confusionTotalRF + confusionRF %adding the value of the confusionRF to the total confusion matrix
                hyperTable = [hyperparRF accuracyRF]; %join all the values in one table
                counterRF = counterRF + 1
            end
            fprintf('this is the confusion matrix number %.2f\n',l)
            confusionRF
            accRF(l) = accuracy_calc(confusionRF);
            confusionTotalRF
            accTotalRF = accuracy_calc(confusionTotalRF)
            counterRF
        end
    end
end
toc
%% Confusion matrix for Random Forest
matRF = confusionchart(confusionTotalRF);
matRF.title('Confusion Matrix for Random Forest')

%% STAGE 3 - RANDOM FOREST - Keeping the Best Hypermeters for our model

[~,idx] = sort(hyperTable(:,4));
hyperTable = hyperTable(idx,:);

%Keeping the best combination of parameters that come from the trained model, based on the the Maximum Accuracy
optimal_hyperparams_RF = hyperTable(end,:);
%%
rng(1) % random seed
% Applying optimal_hyperparams_RF = 40 number of trees to the best hyperparameter values to train the final random forest model
% and avoid overtraining as well, based on the OOB Error plot created.
final_optimal_modelRF = TreeBagger(optimal_hyperparams_RF(1),X1,Y1,'OOBPrediction', 'on', 'minLeafSize', optimal_hyperparams_RF(2), ...
    'NumPredictorsToSample', optimal_hyperparams_RF(3));
[labels,probabilities] = predict(final_optimal_modelRF,X2);
cross_labels_optimal_RF = Y2;
cross_scores_optimal_RF = probabilities(:,2);
labels=str2double(labels);
final_optimal_confusionRF = confusionmat(Y2,labels);
final_optimal_accRF = accuracy_calc(final_optimal_confusionRF);
final_optimal_accRF
%% 
%Ploting the OOB Error curve for 320 Trees
errorsBest = [];
rng(1) %for reproducibility issues
optimal_modelRF = TreeBagger(320,X1,Y1,'OOBPrediction', 'on', 'minLeafSize', optimal_hyperparams_RF(2), ...
    'NumPredictorsToSample', optimal_hyperparams_RF(3));
% Calculating the OOB error based on the optimal model's number of trees
errorsBest = oobError(optimal_modelRF);
% Ploting the OOB error for each number of 320 trees
figure;
plot(errorsBest)
xlabel 'Number of trees';
ylabel 'Out of bag classification error';
title('Out of Bag Classification Error');

rng(1) % random seed
% Applying 160 number of trees to the best hyperparameter values to train the final random forest model
% and avoid overtraining as well, based on the plot created before.
final_optimal_modelRF = TreeBagger(160,X1,Y1,'OOBPrediction', 'on', 'minLeafSize', optimal_hyperparams_RF(2), ...
    'NumPredictorsToSample', optimal_hyperparams_RF(3));
[labels,probabilities] = predict(final_optimal_modelRF,X2);
cross_labels_optimal_RF = Y2;
cross_scores_optimal_RF = probabilities(:,2);
labels=str2double(labels);
final_optimal_confusionRF = confusionmat(Y2,labels);
final_optimal_accRF = accuracy_calc(final_optimal_confusionRF);
final_optimal_accRF
%%
%ROC Curve for Random Forest using the perfcurve function.
[FP,TP,T,AUC, OPTROCPT]= perfcurve(cross_labelsRF,cross_scoresRF,2);
fprintf('AUC = %f\n',AUC);
figure
plot(FP,TP)
hold on
plot((0:0.1:1), (0:0.1:1));
hold off
legend('Type 1', 'Type 2', 'Type 3');
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Random Forest')


%ROC Curve for Random Forest using the perfcurve function with the Optimal
%model
[FP,TP,T,AUC, OPTROCPT]= perfcurve(cross_labels_optimal_RF,cross_scores_optimal_RF,2);
fprintf('AUC = %f\n',AUC);
figure
plot(FP,TP)
hold on
plot((0:0.1:1), (0:0.1:1));
hold off
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Random Forest Optimal Hyperparameters')

%% %
%Calculating the accuracy function

function accuracy = accuracy_calc(confusion_m) %calculating the accuracy of our model based on the confusion matrix
sum = 0;
diag = 0;
for i=1:3
    for j=1:3
        sum = sum+confusion_m(i,j); %summing all the values of our matrix
    end
    diag = diag+confusion_m(i,i); %summing the values of the diagonal
end
accuracy = (diag/sum)*100;
end
