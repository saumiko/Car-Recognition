FileData = load('cars_meta.mat');
T = struct2table(FileData);
writetable(T,'cars_meta.csv','Delimiter',',');

FileData = load('cars_test_annos.mat');
T = struct2table(cell2mat(struct2cell(FileData)));
writetable(T,'cars_test_annos.csv','Delimiter',',');

FileData = load('cars_train_annos.mat');
T = struct2table(cell2mat(struct2cell(FileData)));
writetable(T,'cars_train_annos.csv','Delimiter',',');