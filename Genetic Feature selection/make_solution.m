function A=make_solution(Nf)
    m=randsrc(1,1,1:0.75*Nf);
    A=randperm(Nf,m);
end