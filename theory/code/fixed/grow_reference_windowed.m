function M_train = grow_reference_windowed(A_trunc, n_end_train)
    % Этап 2 (формулы (2)-(3) Вариант_4.docx) — буквальный порт "второго
    % этапа" DATA_100.m: старт с первого вектора усечённого множества,
    % затем n_end_train-1 итераций жадного добавления вектора с минимальным
    % R(x, текущий базис) среди кандидатов НАЧИНАЯ С позиции Nklass
    % (позиционно ограниченный пул — оригинальная особенность алгоритма,
    % не ошибка; см. center_strategy="windowed" в subspace_conjugacy.algorithms).
    n_trunc = size(A_trunc, 2);
    M_train = A_trunc(:, 1);
    for Nklass = 1:(n_end_train - 1)
        R_start = 1.0;
        V_min = [];
        XtX_inv = inv(M_train' * M_train);
        for k_test = Nklass:n_trunc
            if k_test == 1
                continue; % уже использован как стартовый столбец
            end
            X_tmp = A_trunc(:, k_test);
            Y_R = M_train' * X_tmp;
            R = (Y_R' * XtX_inv * Y_R) / (X_tmp' * X_tmp);
            if R < R_start
                R_start = R;
                V_min = X_tmp;
            end
        end
        M_train = [M_train, V_min];
    end
end
