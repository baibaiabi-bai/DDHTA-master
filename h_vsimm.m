function [Y]=h_vsimm(X)
    xEast=X(1);
    yNorth=X(4);
    zUp=X(7);
    
    
    [az,elev,slantRange] = enu2aer(xEast,yNorth,zUp);
    R=slantRange;
    theta=az;
    fai=elev;
    
    Y=[theta fai R]';
end