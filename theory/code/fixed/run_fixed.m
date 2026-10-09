% run_fixed.m — исправленная и работоспособная версия theory/code/Run_200.m,
% запускаемая на реальном центрированном датасете datasets/{class}_centered/.
%
% Исправленные недочёты оригинала (Run_200.m, DATA_100.m, Otbor_100.m,
% DATA_200.m, Recogn_VSPU_S1.m — см. полный разбор в предыдущем анализе):
%
%   1. КРАХ: Run_200.m вызывал Otbor_100() с необъявленной переменной
%      M_test_Kor_3 — гарантированная ошибка на этой строке, ДО того как
%      скрипт доходит до классификации. При этом уже посчитанный (корректный)
%      результат DATA_100 тут же отбрасывался (мёртвый код). Здесь вызов
%      Otbor_100 убран, используется напрямую результат DATA_100 (единственная
%      в оригинале ПОЛНАЯ, двухэтапная реализация отбора).
%
%   2. ПРОИЗВОДИТЕЛЬНОСТЬ: этап 1 DATA_100.m (поиск максимально
%      коррелированной пары) делал ~1.17 млн отдельных вызовов corrcoef()
%      на класс (тройной вложенный цикл) — при типичной цене вызова это
%      от 15 минут до полутора часов НА КЛАСС. Заменено на ОДНОКРАТНЫЙ
%      расчёт полной корреляционной матрицы corrcoef(A) (одна BLAS-операция)
%      + поиск максимума по уже готовой матрице на каждой итерации удаления.
%      Результат идентичен (корреляция пары не меняется от удаления других
%      векторов), но выполняется за секунды, а не за час.
%
%   3. ПРОИЗВОДИТЕЛЬНОСТЬ: DATA_200.m векторизовало каждое изображение
%      ручным двойным циклом (256×256 скалярных присваиваний). Заменено на
%      reshape(img.', [], 1) — тот же порядок развёртки (по строкам),
%      совпадает с vectorize_image(method="horizontal") библиотеки
%      subspace_conjugacy.
%
%   4. УТЕЧКА ДАННЫХ (главный методологический недочёт, уже задокументирован
%      в README проекта): оригинал вызывает Recogn_VSPU_S1(M_test_3, M_train_3),
%      где M_train_3 — эталонные векторы, отобранные ИЗ ТОГО ЖЕ M_test_3, на
%      котором затем измеряется точность. Здесь референсные векторы отбираются
%      ТОЛЬКО из train-пула (160 из 200 на класс, честный стратифицированный
%      80/20), а точность измеряется на ОТДЕЛЬНОМ, непересекающемся test-наборе
%      (оставшиеся 40 на класс) — методология evaluation.paper_train_test_split
%      библиотеки. Ниже приводятся ОБА числа: честное (fixed split) и
%      воспроизведение оригинальной (утечка train==test пул) методологии —
%      чтобы явно показать цену утечки, как во всех остальных экспериментах
%      этого проекта.
%
%   5. Recogn_VSPU_S1.m сглаживало соседние ПО ИНДЕКСУ тестовые векторы
%      окном 3/5/7 — это осмысленно только для последовательных срезов
%      одного пациента, а не для случайно перемешанного набора картинок
%      разных пациентов (см. docstring smooth_adjacent_vectors библиотеки:
%      предпосылка о содержательном порядке не гарантируется и здесь не
%      выполняется). Сглаживание убрано.
%
%   6. Решающее правило использовало хардкод Index(3) ("3-й класс после
%      сортировки по возрастанию R") — работает только при РОВНО 3 классах
%      и не показывает, что это argmax. Заменено на явный [~, best] = max(R_sum).
%
% Алгоритм отбора (этапы 1 и 2, формулы (1)-(3) theory/state_v2/Вариант_4.docx)
% воспроизведён БУКВАЛЬНО, включая позиционно-ограниченный ("windowed") пул
% кандидатов на этапе 2 — это не бага, а особенность оригинального алгоритма
% (в python-библиотеке она сознательно сохранена как опция center_strategy=
% "windowed", см. README, раздел "Опциональные находки из MATLAB-кода").
%
% Запуск (из корня репозитория):
%   octave-cli theory/code/fixed/run_fixed.m

clear; close all; more off;
addpath(fileparts(mfilename('fullpath'))); % truncate_by_correlation.m, grow_reference_windowed.m, classify_and_evaluate.m
t_total = tic;

N_K = 256; N_L = 256; N_Vektor = N_K * N_L;
CLASSES = {'glioma', 'meningioma', 'pituitary'};
N_CLASSES = numel(CLASSES);
N_PER_CLASS = 200;
N_END_TRAIN = 10;      % как в оригинальном Run_200.m (N_end_train)
SEED = 42;

TRAIN_POOL_SIZE = 160; % честный 80/20 сплит (как paper_train_test_split)
TEST_SIZE = 40;
TRUNCATED_HONEST = 80; % этап 1: тот же коэффициент отбрасывания половины, что в оригинале (200->100)

TRUNCATED_LEAKY = 100;  % воспроизведение оригинальной (утечка) методологии на полном пуле 200

printf('=== run_fixed.m: датасет = datasets/{class}_centered, N_END_TRAIN=%d ===\n', N_END_TRAIN);

% ------------------------------------------------------------------
% Загрузка и векторизация (исправление п.3 — reshape вместо двойного цикла)
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
            img = img(:, :, 1); % каналы идентичны (grayscale, сохранён как RGB PNG)
        end
        imgd = double(img) / 255;
        Xc(:, k) = reshape(imgd.', [], 1); % row-major развёртка, как vectorize_image(method="horizontal")
    end
    X_by_class{c} = Xc;
    printf('   %s: %d векторов загружено\n', cls, size(Xc, 2));
end
printf('готово за %.1f с\n', toc(t0));

% ------------------------------------------------------------------
% Честный 80/20 сплит на класс (исправление п.4 — устранение утечки)
% ------------------------------------------------------------------
rand('seed', SEED);
train_pool_by_class = cell(1, N_CLASSES);
test_by_class = cell(1, N_CLASSES);
split_manifest = struct();
for c = 1:N_CLASSES
    cls = CLASSES{c};
    d = dir(fullfile('datasets', [cls '_centered'], '*.png'));
    names = sort({d.name});
    names = names(1:N_PER_CLASS);
    perm = randperm(N_PER_CLASS);
    train_idx = perm(1:TRAIN_POOL_SIZE);
    test_idx = perm(TRAIN_POOL_SIZE + 1:end);
    train_pool_by_class{c} = X_by_class{c}(:, train_idx);
    test_by_class{c} = X_by_class{c}(:, test_idx);
    split_manifest.(cls).train = names(train_idx);
    split_manifest.(cls).test = names(test_idx);
end
% Экспорт точного разбиения (для воспроизведения ТЕХ ЖЕ изображений в
% Python — RNG Octave и numpy/sklearn несовместимы даже при одном "seed").
json_str = jsonencode(split_manifest);
fid = fopen('artifacts/matlab_variant_b_split.json', 'w');
fputs(fid, json_str);
fclose(fid);
printf('Сплит Варианта B экспортирован в artifacts/matlab_variant_b_split.json\n');

% ==================================================================
% ВАРИАНТ A — воспроизведение ОРИГИНАЛЬНОЙ методологии Run_200.m
% (референсы отбираются из ВСЕХ 200, точность измеряется на тех же 200 —
% утечка данных, оставлена намеренно для прямого сравнения с честным B)
% ==================================================================
printf('\n=== Вариант A: оригинальная методология (train==test пул, утечка) ===\n');
t0 = tic;
M_train_leaky = cell(1, N_CLASSES);
for c = 1:N_CLASSES
    kept_idx = truncate_by_correlation(X_by_class{c}, TRUNCATED_LEAKY);
    A_trunc = X_by_class{c}(:, kept_idx);
    M_train_leaky{c} = grow_reference_windowed(A_trunc, N_END_TRAIN);
    printf('   %s: усечено 200->%d, эталонов отобрано %d\n', ...
           CLASSES{c}, TRUNCATED_LEAKY, size(M_train_leaky{c}, 2));
end
[acc_leaky, per_class_leaky] = classify_and_evaluate(M_train_leaky, X_by_class);
printf('Точность (Вариант A, утечка), по классам:\n');
for c = 1:N_CLASSES
    printf('   %s: %.4f\n', CLASSES{c}, per_class_leaky(c));
end
printf('Средняя точность (Вариант A, утечка): %.4f (за %.1f с)\n', acc_leaky, toc(t0));

% ==================================================================
% ВАРИАНТ B — честная методология (train pool / test независимы)
% ==================================================================
printf('\n=== Вариант B: честная методология (train pool 160 / test 40, без утечки) ===\n');
t0 = tic;
M_train_honest = cell(1, N_CLASSES);
for c = 1:N_CLASSES
    kept_idx = truncate_by_correlation(train_pool_by_class{c}, TRUNCATED_HONEST);
    A_trunc = train_pool_by_class{c}(:, kept_idx);
    M_train_honest{c} = grow_reference_windowed(A_trunc, N_END_TRAIN);
    printf('   %s: усечено 160->%d, эталонов отобрано %d\n', ...
           CLASSES{c}, TRUNCATED_HONEST, size(M_train_honest{c}, 2));
end
[acc_honest, per_class_honest] = classify_and_evaluate(M_train_honest, test_by_class);
printf('Точность (Вариант B, честно), по классам:\n');
for c = 1:N_CLASSES
    printf('   %s: %.4f\n', CLASSES{c}, per_class_honest(c));
end
printf('Средняя точность (Вариант B, честно): %.4f (за %.1f с)\n', acc_honest, toc(t0));

printf('\n=== ИТОГ ===\n');
printf('Вариант A (train==test, утечка, N_end_train=%d): mean accuracy = %.4f\n', N_END_TRAIN, acc_leaky);
printf('Вариант B (train/test независимы, N_end_train=%d): mean accuracy = %.4f\n', N_END_TRAIN, acc_honest);
printf('Цена утечки (A - B): %+.4f\n', acc_leaky - acc_honest);
printf('\nВсего времени: %.1f с\n', toc(t_total));

% Локальные функции вынесены в отдельные файлы этой же директории
% (truncate_by_correlation.m, grow_reference_windowed.m,
% classify_and_evaluate.m) — Octave/MATLAB надёжно поддерживают функции
% "один файл = одна функция", в отличие от локальных функций в конце
% скрипт-файла при запуске в batch-режиме.
