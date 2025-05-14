function acc = fitf(sub)
    global orgfeatures labels alg
    features = orgfeatures(:, sub);
    if isempty(features) || all(sub == 0)
        acc = 0; % Return a low fitness if no features are selected
    else
        acc = evl(features, labels, alg); % Your existing evaluation function
    end
end
