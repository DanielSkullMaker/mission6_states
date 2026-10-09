function keep_idx = truncate_by_correlation(X, target_count)
    % Этап 1 (формула (1) Вариант_4.docx): повторяет "удалить один вектор
    % из максимально коррелированной пары" до тех пор, пока не останется
    % target_count векторов. Исправление недочёта DATA_100.m: полная
    % корреляционная матрица считается ОДИН раз (corrcoef — одна
    % векторизованная BLAS-операция), а не через ~n^2/2 отдельных вызовов
    % corrcoef() в цикле — результат идентичен оригиналу, т.к. корреляция
    % пары (i,j) не зависит от того, какие ДРУГИЕ векторы ещё присутствуют
    % в рабочем множестве.
    n = size(X, 2);
    R = corrcoef(X);
    R(1:n+1:end) = -Inf; % убираем диагональ из рассмотрения
    alive = true(n, 1);
    n_alive = n;
    while n_alive > target_count
        Rmask = R;
        Rmask(~alive, :) = -Inf;
        Rmask(:, ~alive) = -Inf;
        [~, lin] = max(Rmask(:));
        [i, j] = ind2sub([n, n], lin);
        remove_idx = max(i, j); % как в оригинале — убирается "j_out" (второй индекс пары)
        alive(remove_idx) = false;
        n_alive = n_alive - 1;
    end
    keep_idx = find(alive)';
end
