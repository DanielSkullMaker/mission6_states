function [mean_acc, per_class_acc] = classify_and_evaluate(M_train_by_class, X_test_by_class)
    % Решающее правило (4) Вариант_4.docx: argmax R(x, Y_класса).
    % Исправление недочёта Recogn_VSPU_S1.m: явный argmax вместо хардкода
    % Index(3). Сглаживание соседних тестовых векторов НЕ применяется
    % (порядок тестовых изображений здесь не последовательный по пациенту,
    % см. docstring smooth_adjacent_vectors библиотеки subspace_conjugacy).
    n_classes = numel(M_train_by_class);
    inv_by_class = cell(1, n_classes);
    for l = 1:n_classes
        Y = M_train_by_class{l};
        inv_by_class{l} = inv(Y' * Y);
    end
    per_class_acc = zeros(1, n_classes);
    total_correct = 0;
    total_n = 0;
    for k = 1:n_classes
        X_test = X_test_by_class{k};
        n_test = size(X_test, 2);
        correct = 0;
        for i = 1:n_test
            xt = X_test(:, i);
            R_sum = zeros(1, n_classes);
            for l = 1:n_classes
                Y = M_train_by_class{l};
                Y_R = Y' * xt;
                R_sum(l) = (Y_R' * inv_by_class{l} * Y_R) / (xt' * xt);
            end
            [~, best] = max(R_sum);
            if best == k
                correct = correct + 1;
            end
        end
        per_class_acc(k) = correct / n_test;
        total_correct = total_correct + correct;
        total_n = total_n + n_test;
    end
    mean_acc = total_correct / total_n;
end
