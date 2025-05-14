% clear all; close all; clc;
% 
% % Load features and labels
% load('bestCCnetfeatureswithlables45fc8.mat');  % features and labels
% 
% % Algorithm parameters
% dim = size(features, 2);
% N = 100;
% iter = 100;
% upbound = max(features, [], 1);
% lowbound = min(features, [], 1);
% 
% % Initialization
% X = bsxfun(@plus, lowbound, bsxfun(@times, (upbound - lowbound), rand(N, dim)));
% V = zeros(N, dim);
% alfa = 0.1;
% G = 9.81;
% 
% % Rainfall Algorithm Execution
% for iteration = 1:iter
%     fitness = sum(X, 2);
%     [Fbest, idx] = min(fitness);
%     Lbest = X(idx, :);
% 
%     for i = 1:N
%         for j = 1:dim
%             V(i, j) = V(i, j) + G * (Lbest(j) - X(i, j)) * exp(-alfa * iteration);
%             X(i, j) = X(i, j) + V(i, j);
%             X(i, j) = min(max(X(i, j), lowbound(j)), upbound(j));
%         end
%     end
% end
% 
% % Feature Selection
% % Adjusting the selection mechanism
% selectedFeatureIndices = X(idx, :) > mean(X(idx, :));  % Dynamic threshold based on mean
% if sum(selectedFeatureIndices) == 0
%     disp('Adjusting threshold due to no features selected.');
%     selectedFeatureIndices = X(idx, :) > min(X(idx, :));  % Lower threshold to minimum value
% end
% 
% selectedFeatures = features(:, selectedFeatureIndices);
% if isempty(selectedFeatures)
%     error('Selected features are empty. Check feature selection criteria and thresholds.');
% end
% 
% % Proceed with classification...
% 
% % Data Partition
% cv = cvpartition(labels, 'HoldOut', 0.3);
% trainIdx = cv.training();
% testIdx = cv.test();
% 
% % Classifiers
% classifiers = {@fitcknn, @fitcecoc, @fitcnb, @fitctree};
% classifierNames = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};
% 
% % Evaluation
% for i = 1:length(classifiers)
%     model = classifiers{i}(selectedFeatures(trainIdx, :), labels(trainIdx));
%     predictions = predict(model, selectedFeatures(testIdx, :));
%     C = confusionmat(labels(testIdx), predictions);
% 
%     figure;
%     confusionchart(C);
%     title([classifierNames{i} ' Confusion Matrix']);
% end



clear all; close all; clc;

% Load features and labels
load('bestCCnetfeatureswithlables45fc8.mat');  % features and labels

% Algorithm parameters
dim = size(features, 2);
N = 100;
iter = 100;
upbound = max(features, [], 1);
lowbound = min(features, [], 1);

% Initialization
X = bsxfun(@plus, lowbound, bsxfun(@times, (upbound - lowbound), rand(N, dim)));
V = zeros(N, dim);
alfa = 0.1;
G = 9.81;

% Rainfall Algorithm Execution
for iteration = 1:iter
    fitness = sum(X, 2);
    [Fbest, idx] = min(fitness);
    Lbest = X(idx, :);

    for i = 1:N
        for j = 1:dim
            V(i, j) = V(i, j) + G * (Lbest(j) - X(i, j)) * exp(-alfa * iteration);
            X(i, j) = X(i, j) + V(i, j);
            X(i, j) = min(max(X(i, j), lowbound(j)), upbound(j));
        end
    end
end

% Feature Selection
% Adjusting the selection mechanism
selectedFeatureIndices = X(idx, :) > mean(X(idx, :));  % Dynamic threshold based on mean
if sum(selectedFeatureIndices) == 0
    disp('Adjusting threshold due to no features selected.');
    selectedFeatureIndices = X(idx, :) > min(X(idx, :));  % Lower threshold to minimum value
end

selectedFeatures = features(:, selectedFeatureIndices);
if isempty(selectedFeatures)
    error('Selected features are empty. Check feature selection criteria and thresholds.');
end

% Data Partition
cv = cvpartition(labels, 'HoldOut', 0.3);
trainIdx = cv.training();
testIdx = cv.test();

% Classifiers
classifiers = {@fitcknn, @fitcecoc, @fitcnb, @fitctree};
classifierNames = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};

% Store results for number of features and classification accuracy
numFeaturesUsed = sum(selectedFeatureIndices);  % All classifiers use the same number of features
classificationResults = zeros(1, length(classifiers));  % To store classification accuracy for each classifier

% Evaluation
for i = 1:length(classifiers)
    % Train the model
    model = classifiers{i}(selectedFeatures(trainIdx, :), labels(trainIdx));
    predictions = predict(model, selectedFeatures(testIdx, :));
    
    % Confusion Matrix
    C = confusionmat(labels(testIdx), predictions);

    % Calculate classification accuracy
    classificationResults(i) = sum(predictions == labels(testIdx)) / length(labels(testIdx));

    % Plot confusion matrix
    figure;
    confusionchart(C);
    title([classifierNames{i} ' Confusion Matrix']);
end

% Plot the relationship between the number of features and classification performance
figure;
bar(classificationResults); % Using classification results (accuracy) as y-values
set(gca, 'XTickLabel', classifierNames); % X-axis as classifiers
xlabel('Classifier');
ylabel('Accuracy');
title('Classification Performance vs. Number of Features');
grid on;

% Display results
disp('Number of features used by each classifier:');
disp([num2str(numFeaturesUsed) ' features used by all classifiers']);

% Show classification performance and number of features
disp('Classification performance (accuracy) for each classifier:');
for i = 1:length(classifiers)
    disp([classifierNames{i} ': Accuracy = ' num2str(classificationResults(i)) ', Features used = ' num2str(numFeaturesUsed)]);
end
