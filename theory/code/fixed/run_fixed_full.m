% run_fixed_full.m — расширение run_fixed.m: та же пара вариантов (A —
% train_pool==test_pool, утечка; B — честный 160/40 сплит), но для ОБОИХ
% стратегий отбора эталонов (критерий (3) и случайный отбор — аналог
% Table 1/Table 2 theory/state_v2/Вариант_4_1.docx / experiments/
% experiment_draft_variant2.py::run_table1/run_table2_random) и для
% m = 10, 20, 30, 40, 50 (не только m=10, как в run_fixed.m).
%
% Экономия вычислений: критерий (3) растится ОДИН раз до m=50 на класс на
% вариант (grow_reference_windowed) — базис для меньших m это просто
% префикс первых m столбцов уже выросшего базиса (последовательный жадный
% рост не меняет уже выбранные векторы при продолжении). Аналогично
% случайный отбор — одна случайная перестановка усечённого пула на класс
% на вариант, префикс первых m элементов на каждое m. Это тот же принцип,
% что и TABLE1_SAMPLE_SIZES в experiment_draft_variant2.py, но для MATLAB
% реализация отбора добавлена заново (её там не было).
%
% Вывод: 4 таблицы (критерий(3)/случайно x A/B) + Таблица 5 (цена утечки
% A-B для каждой стратегии отдельно) + JSON-экспорт для сверки с Python.
%
% Запуск (из корня репозитория):
%   octave-cli theory/code/fixed/run_fixed_full.m

clear; close all; more off;
addpath(fileparts(mfilename('fullpath')));
t_total = tic;

N_K = 256; N_L = 256; N_Vektor = N_K * N_L;
CLASSES = {'glioma', 'meningioma', 'pituitary'};
N_CLASSES = numel(CLASSES);
N_PER_CLASS = 200;
M_VALUES = [10, 20, 30, 40, 50];
M_MAX = max(M_VALUES);
N_M = numel(M_VALUES);
SEED = 42;

TRAIN_POOL_SIZE = 160;
TRUNCATED_HONEST = 80;  % Вариант B: 160 -> 80 (тот же коэффициент 1/2, что в оригинале)
TRUNCATED_LEAKY = 100;  % Вариант A: 200 -> 100 (как в DATA_100.m/оригинале)

printf('=== run_fixed_full.m: M_VALUES=[%s] ===\n', num2str(M_VALUES));

% ------------------------------------------------------------------
% Загрузка и векторизация (как в run_fixed.m)
% ------------------------------------------------------------------
printf('\nЗагрузка и векторизация %d x %d изображений...\n', N_PER_CLASS, N_CLASSES);
t0 = tic;
X_by_class = cell(1, N_CLASSES);
for c = 1:N_CLASSES
    cls = CLASSES{c};
    d = dir(fullfile('datasets', [cls '_centered'], '*.png'));
    names = sort({d.name});
    if numel(names) < N_PER_CLASS
        error('Класс %s: найдено %d файлов, требуется %d.', cls, numel(names), N_PER_CLASS);
    end
    names = names(1:N_PER_CLASS);
    Xc = zeros(N_Vektor, N_PER_CLASS);
    for k = 1:N_PER_CLASS
        img = imread(fullfile('datasets', [cls '_centered'], names{k}));
        if ndims(img) == 3
            img = img(:, :, 1);
        end
        imgd = double(img) / 255;
        Xc(:, k) = reshape(imgd.', [], 1);
    end
    X_by_class{c} = Xc;
    printf('   %s: %d векторов загружено\n', cls, size(Xc, 2));
end
printf('готово за %.1f с\n', toc(t0));

% ------------------------------------------------------------------
% Честный 80/20 сплит на класс (для Варианта B) — тот же SEED, что и в
% run_fixed.m, поэтому даёт тот же train_pool/test_by_class.
% ------------------------------------------------------------------
rand('seed', SEED);
train_pool_by_class = cell(1, N_CLASSES);
test_by_class = cell(1, N_CLASSES);
for c = 1:N_CLASSES
    perm = randperm(N_PER_CLASS);
    train_pool_by_class{c} = X_by_class{c}(:, perm(1:TRAIN_POOL_SIZE));
    test_by_class{c} = X_by_class{c}(:, perm(TRAIN_POOL_SIZE + 1:end));
end

% Результаты: acc_<strategy>_<variant>(mi), pc_<strategy>_<variant>(mi, 1:3)
acc_crit3_A = zeros(1, N_M); pc_crit3_A = zeros(N_M, N_CLASSES);
acc_random_A = zeros(1, N_M); pc_random_A = zeros(N_M, N_CLASSES);
acc_crit3_B = zeros(1, N_M); pc_crit3_B = zeros(N_M, N_CLASSES);
acc_random_B = zeros(1, N_M); pc_random_B = zeros(N_M, N_CLASSES);

% ==================================================================
% ВАРИАНТ A — train_pool == test_pool == все 200/класс (утечка)
% ==================================================================
printf('\n=== Вариант A (усечение 200->%d/класс, утечка) ===\n', TRUNCATED_LEAKY);
t0 = tic;
crit3_full_A = cell(1, N_CLASSES);
random_full_A = cell(1, N_CLASSES);
rand('seed', SEED + 100);  % отдельный сид для случайного отбора (не пересекается со сплитом выше)
for c = 1:N_CLASSES
    kept_idx = truncate_by_correlation(X_by_class{c}, TRUNCATED_LEAKY);
    A_trunc = X_by_class{c}(:, kept_idx);
    crit3_full_A{c} = grow_reference_windowed(A_trunc, M_MAX);
    n_trunc = size(A_trunc, 2);
    perm = randperm(n_trunc);
    random_full_A{c} = A_trunc(:, perm(1:M_MAX));
    printf('   %s: усечено 200->%d, базисы до m=%d построены (крит.3 и случайно)\n', ...
           CLASSES{c}, TRUNCATED_LEAKY, M_MAX);
end
printf('   отбор для Варианта A готов за %.1f с\n', toc(t0));

for mi = 1:N_M
    m = M_VALUES(mi);
    M_crit3 = cell(1, N_CLASSES);
    M_random = cell(1, N_CLASSES);
    for c = 1:N_CLASSES
        M_crit3{c} = crit3_full_A{c}(:, 1:m);
        M_random{c} = random_full_A{c}(:, 1:m);
    end
    [acc_crit3_A(mi), pc] = classify_and_evaluate(M_crit3, X_by_class);
    pc_crit3_A(mi, :) = pc;
    [acc_random_A(mi), pc] = classify_and_evaluate(M_random, X_by_class);
    pc_random_A(mi, :) = pc;
    printf('   m=%2d: критерий(3)=%.4f  случайно=%.4f\n', m, acc_crit3_A(mi), acc_random_A(mi));
end

% ==================================================================
% ВАРИАНТ B — train_pool 160 / test 40 на класс (честно, независимо)
% ==================================================================
printf('\n=== Вариант B (усечение 160->%d/класс, честно) ===\n', TRUNCATED_HONEST);
t0 = tic;
crit3_full_B = cell(1, N_CLASSES);
random_full_B = cell(1, N_CLASSES);
rand('seed', SEED + 100);
for c = 1:N_CLASSES
    kept_idx = truncate_by_correlation(train_pool_by_class{c}, TRUNCATED_HONEST);
    A_trunc = train_pool_by_class{c}(:, kept_idx);
    crit3_full_B{c} = grow_reference_windowed(A_trunc, M_MAX);
    n_trunc = size(A_trunc, 2);
    perm = randperm(n_trunc);
    random_full_B{c} = A_trunc(:, perm(1:M_MAX));
    printf('   %s: усечено 160->%d, базисы до m=%d построены (крит.3 и случайно)\n', ...
           CLASSES{c}, TRUNCATED_HONEST, M_MAX);
end
printf('   отбор для Варианта B готов за %.1f с\n', toc(t0));

for mi = 1:N_M
    m = M_VALUES(mi);
    M_crit3 = cell(1, N_CLASSES);
    M_random = cell(1, N_CLASSES);
    for c = 1:N_CLASSES
        M_crit3{c} = crit3_full_B{c}(:, 1:m);
        M_random{c} = random_full_B{c}(:, 1:m);
    end
    [acc_crit3_B(mi), pc] = classify_and_evaluate(M_crit3, test_by_class);
    pc_crit3_B(mi, :) = pc;
    [acc_random_B(mi), pc] = classify_and_evaluate(M_random, test_by_class);
    pc_random_B(mi, :) = pc;
    printf('   m=%2d: критерий(3)=%.4f  случайно=%.4f\n', m, acc_crit3_B(mi), acc_random_B(mi));
end

% ------------------------------------------------------------------
% Печать 4 таблиц + Таблица 5 (цена утечки)
%
% Без локальной функции print_table() — намеренно: local-функции ПОСЛЕ
% которых идёт ещё скриптовый код ненадёжны в batch-режиме Octave (см.
% комментарий в конце run_fixed.m про "один файл = одна функция"), поэтому
% каждая из 4 таблиц печатается инлайн-циклом, без абстракции.
% ------------------------------------------------------------------
printf('\n=== Таблица 1-A: критерий (3), Вариант A (train==test, утечка) ===\n');
printf('%4s | %8s | %10s | %9s | %8s\n', 'm', 'glioma', 'meningioma', 'pituitary', 'mean');
for i = 1:N_M
    printf('%4d | %8.4f | %10.4f | %9.4f | %8.4f\n', ...
           M_VALUES(i), pc_crit3_A(i,1), pc_crit3_A(i,2), pc_crit3_A(i,3), acc_crit3_A(i));
end

printf('\n=== Таблица 1-B: критерий (3), Вариант B (честный 160/40 сплит) ===\n');
printf('%4s | %8s | %10s | %9s | %8s\n', 'm', 'glioma', 'meningioma', 'pituitary', 'mean');
for i = 1:N_M
    printf('%4d | %8.4f | %10.4f | %9.4f | %8.4f\n', ...
           M_VALUES(i), pc_crit3_B(i,1), pc_crit3_B(i,2), pc_crit3_B(i,3), acc_crit3_B(i));
end

printf('\n=== Таблица 2-A: случайный отбор, Вариант A (train==test, утечка) ===\n');
printf('%4s | %8s | %10s | %9s | %8s\n', 'm', 'glioma', 'meningioma', 'pituitary', 'mean');
for i = 1:N_M
    printf('%4d | %8.4f | %10.4f | %9.4f | %8.4f\n', ...
           M_VALUES(i), pc_random_A(i,1), pc_random_A(i,2), pc_random_A(i,3), acc_random_A(i));
end

printf('\n=== Таблица 2-B: случайный отбор, Вариант B (честный 160/40 сплит) ===\n');
printf('%4s | %8s | %10s | %9s | %8s\n', 'm', 'glioma', 'meningioma', 'pituitary', 'mean');
for i = 1:N_M
    printf('%4d | %8.4f | %10.4f | %9.4f | %8.4f\n', ...
           M_VALUES(i), pc_random_B(i,1), pc_random_B(i,2), pc_random_B(i,3), acc_random_B(i));
end

printf('\n=== Таблица 5: Цена утечки (Вариант A - Вариант B) ===\n');
printf('%4s | %14s | %14s\n', 'm', 'критерий (3)', 'случайный отбор');
for i = 1:N_M
    printf('%4d | %+14.4f | %+14.4f\n', M_VALUES(i), ...
           acc_crit3_A(i) - acc_crit3_B(i), acc_random_A(i) - acc_random_B(i));
end

% ------------------------------------------------------------------
% JSON-экспорт
% ------------------------------------------------------------------
report = struct();
report.m_values = M_VALUES;
report.table1_A_criterion3 = struct('mean', acc_crit3_A, 'per_class', pc_crit3_A);
report.table1_B_criterion3 = struct('mean', acc_crit3_B, 'per_class', pc_crit3_B);
report.table2_A_random = struct('mean', acc_random_A, 'per_class', pc_random_A);
report.table2_B_random = struct('mean', acc_random_B, 'per_class', pc_random_B);
report.leakage_price_criterion3 = acc_crit3_A - acc_crit3_B;
report.leakage_price_random = acc_random_A - acc_random_B;

json_str = jsonencode(report);
fid = fopen('artifacts/matlab_variant_ab_tables.json', 'w');
fputs(fid, json_str);
fclose(fid);
printf('\nJSON-отчёт: artifacts/matlab_variant_ab_tables.json\n');

printf('\nВсего времени: %.1f с\n', toc(t_total));
