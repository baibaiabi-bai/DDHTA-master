function Mkc_seq=get_Mkc_seq(a,b)
c=[];
for i = 1:length(a)
    
    if ~ismember(a(i), b)
        c = [c a(i)];
    end
end
Mkc_seq=c;
end