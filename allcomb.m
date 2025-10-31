function result = allcomb(varargin)
    args = varargin;
    n = nargin;
    [F{1:n}] = ndgrid(args{:});
    for i=n:-1:1
        G(:,i) = F{i}(:);
    end
    result = G;
end