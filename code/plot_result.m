BW = load('BW.mat');
TF = load('TF.mat');
WV = load('WV.mat');

x = [200 1000 2000 4000 7000];

figure(1),
data = BW.detail';
plot(x,data(1,:),'b','LineWidth',3)
hold on
plot(x,data(2,:),'k','LineWidth',3)
plot(x,data(3,:),'r','LineWidth',3)
plot(x,data(4,:),'g','LineWidth',3)
plot(x,data(5,:),'y','LineWidth',3)
plot(x,data(6,:),'c','LineWidth',3)
plot(x,data(7,:),'m','LineWidth',3)
hold off
ylabel('Root Mean Square Error')
legend('KNN', 'SVM (rbf)', 'Linear', 'Tree', 'Random Forest', 'MLP', 'AdaBoost', 'Location', 'SouthEast')
title('Bag of Words')

figure(2),
data = TF.detail';
plot(x,data(1,:),'b','LineWidth',3)
hold on
plot(x,data(2,:),'k','LineWidth',3)
plot(x,data(3,:),'r','LineWidth',3)
plot(x,data(4,:),'g','LineWidth',3)
plot(x,data(5,:),'y','LineWidth',3)
plot(x,data(6,:),'c','LineWidth',3)
plot(x,data(7,:),'m','LineWidth',3)
hold off
ylabel('Root Mean Square Error')
legend('KNN', 'SVM (rbf)', 'Linear', 'Tree', 'Random Forest', 'MLP', 'AdaBoost', 'Location', 'SouthEast')
title('Tf-idf')

figure(3),
data = WV.detail';
plot(x,data(1,:),'b','LineWidth',3)
hold on
plot(x,data(2,:),'k','LineWidth',3)
plot(x,data(3,:),'r','LineWidth',3)
plot(x,data(4,:),'g','LineWidth',3)
plot(x,data(5,:),'y','LineWidth',3)
plot(x,data(6,:),'c','LineWidth',3)
plot(x,data(7,:),'m','LineWidth',3)
hold off
ylabel('Root Mean Square Error')
legend('KNN', 'SVM (rbf)', 'Linear', 'Tree', 'Random Forest', 'MLP', 'AdaBoost', 'Location', 'SouthEast')
title('Word2vec')