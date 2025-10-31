function Mpk_seq=get_Mpk(Mk_1,mu,flag)
%%%%%%
[~, max_indices] = maxk(mu,3);
if(flag==0)
Mpk_seq=[Mk_1(max_indices(1)),Mk_1(max_indices(2)),Mk_1(max_indices(3))];
end
if(flag==1)
Mpk_seq=[Mk_1(max_indices(2)),Mk_1(max_indices(3))];
end

end 