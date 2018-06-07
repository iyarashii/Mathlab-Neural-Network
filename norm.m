format compact
P = data(:,1:11)';
T = data(:,12)';

maxP = max(P');
minP = min(P');
Pn = zeros(size(P));
for i = 1:length(maxP),
    Pn(i,:) = (1-(-1))*(P(i,:)-minP(i))/(maxP(i)-minP(i)) + (-1);
%     (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
end

save wino_norm