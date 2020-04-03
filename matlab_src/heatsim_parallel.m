close all;
clc;

tic

initTempRange = 15;
finalTempRange = 40:5:60;
riseTempRange = 15;
specificHeatRange = [921, 385, 130];%, 460, 380];
thermalConductivityRange = [150, 388, 318];%, 59, 151 ];
massDensityRange = [2.70*1000, 8.933*1000, 18.9*1000];%, 7.849*1000, 8.8*1000];
materialRange = ["Al", "Cu", "Au",];% "Fe", "Brass"];
% initTempRange = 15;
% finalTempRange = 60;
% riseTempRange = 15;
% specificHeatRange = [921];
% thermalConductivityRange = [150];
% massDensityRange = [2.70*1000];
% materialRange = ["Al"];

iteration_array = cell(length(initTempRange), length(finalTempRange), length(riseTempRange), length(specificHeatRange), length(thermalConductivityRange), length(massDensityRange), length(materialRange));
iterations = size(iteration_array);

delete(gcp('nocreate'))
parpool('local',4);

for rad = 3:0.5:6   
    
    thermalmodelT = createpde('thermal','transient');
    
    air = rad + 0.05;
    %define the shape of the 'air'
    r1 = [-air air air -air -air -air air air];
    r1 = [3 4 r1./100]';
    %define the shape of the 2d sqaure representing a cylinder
    r2 = [-rad rad rad -rad  -rad -rad rad rad];
    r2 = [3 4 r2./100]';
    gm = [r1,r2];
    sf = 'r1+r2'; %add the shapes together
    ns = char('r1','r2');
    ns = ns';
    [dl,bt] = decsg(gm,sf,ns);
    geometryFromEdges(thermalmodelT,dl); %thermal model is now a
    
    parfor (i = 1:prod(iterations))
%     for (i = 1:prod(iterations))
        [initTempIter, finalTempIter, riseTempIter, specHeatIter, thermCondIter, massDensIter, matIter] = ind2sub(iterations,i);
        
        inittemp = initTempRange(initTempIter);
        risetemp = riseTempRange(riseTempIter);
        finaltemp = finalTempRange(finalTempIter);
        thermalConductivity = thermalConductivityRange(thermCondIter);
        massDensity = massDensityRange(massDensIter);
        specificHeat = specificHeatRange(specHeatIter);
        material = materialRange(matIter);
        
        thermalProperties(thermalmodelT,'Face',1,'ThermalConductivity',25.72/1000,... %W/m K
            'MassDensity',1.2041 ,... %kg/m^3
            'SpecificHeat', 0.718/1000); %J/(kmol*K)
        thermalProperties(thermalmodelT,'Face',2,'ThermalConductivity',thermalConductivity,... %W/m K
            'MassDensity',massDensity,... %kg/m^3
            'SpecificHeat',specificHeat); %J/(kg k)
        
        k_dc = finaltemp-inittemp;
        overshoot = 13;
        settling_time = 200;
        zeta = sqrt(log(overshoot/100)^2/(pi^2 + log(overshoot/100)^2));
        w_n = 4/(settling_time*zeta);
        [~,den] = ord2(w_n, zeta);
        num = k_dc*(w_n)^2;
        [A,B,C,D] = tf2ss(num,den);
        sys = ss(A,B,C,D);
        %         [yi,ti,xi] = initial(sys,[inittemp,0],60*60);
        %         sys = tf(num,den);
        [ys,~,~] = step(sys);
        airtemp = inittemp + ys;
        
        transientBCHeatedBlockParallel(thermalmodelT, 'Edge', [2,1,7,6], 'Temperature', inittemp, risetemp, finaltemp)

        msh= generateMesh(thermalmodelT,'Hmax',0.001);
        %             figure
        %             pdeplot(thermalmodelT);
        %             axis equal
        %             title 'Block With Finite Element Mesh Displayed'
        
        tlist = 0:1:(60*60);
        thermalIC(thermalmodelT,inittemp);
        R = solve(thermalmodelT,tlist);
        T = R.Temperature;
        
        getClosestNode = @(p,x,y) min((p(1,:) - x).^2 + (p(2,:) - y).^2);
        
        [~,nid] = getClosestNode( msh.Nodes, 0, 0 );
        [~,nid2] = getClosestNode( msh.Nodes, 1.5,1.5);
        
        centerT = T(nid,:);
        outT = T(nid2,:);
        specificHeatCSV = zeros(length(tlist),1) + specificHeat;
        thermalConductivityCSV = zeros(length(tlist),1) + thermalConductivity;
        massDensityCSV = zeros(length(tlist),1) + massDensity;
        radCSV = zeros(length(tlist),1) + rad;
        
        csvName = sprintf("%.2f%sTemps%d%d%d.csv", rad, material, inittemp,risetemp,finaltemp)
        csvMat = [tlist', outT', centerT', radCSV, radCSV, specificHeatCSV, thermalConductivityCSV, massDensityCSV];
        writematrix(csvMat,csvName);
 
        %     save(Namecenter,'centerT')
        %     save(Nameouter, 'outT')
    end
    
end
toc