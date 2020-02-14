close all;
clc;

thermalmodelT = createpde('thermal','transient');
global inittemp
global risetemp
global finaltemp
 %define the shape of the 'air'
r1 = [-1.55 1.55 1.55 -1.55 -1.55 -1.55 1.55 1.55];
r1 = [3 4 r1./100]';
%define the shape of the 2d sqaure representing a cylinder
r2 = [-1.5 1.5 1.5 -1.5  -1.5 -1.5 1.5 1.5];
r2 = [3 4 r2./100]';
gm = [r1,r2];
sf = 'r1+r2'; %add the shapes together
ns = char('r1','r2');
ns = ns';
[dl,bt] = decsg(gm,sf,ns);
geometryFromEdges(thermalmodelT,dl); %thermal model is now a 
figure
pdegplot(thermalmodelT,'EdgeLabels','on','FaceLabels','on');

for inital_temp = 15:5:25
    for final_temp = 80:5:100
        for rise_time = 10:5:40
            inittemp = inital_temp
            risetemp = rise_time
            finaltemp = final_temp
            
            thermalProperties(thermalmodelT,'Face',1,'ThermalConductivity',25.72/1000,... %W/m K
                                            'MassDensity',1.2041 ,... %kg/m^3
                                            'SpecificHeat', 0.718/1000); %J/(kmol*K)
            thermalProperties(thermalmodelT,'Face',2,'ThermalConductivity',150,... %W/m K
                                            'MassDensity',2.70*1000,... %kg/m^3
                                            'SpecificHeat',921); %J/(kg k)

            thermalBC(thermalmodelT,'Edge',[2,1,7,6],'Temperature',@transientBCHeatedBlock);

             msh = generateMesh(thermalmodelT,'Hmax',0.001);
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

%            h = figure;
%            h.Position = [1 1 2 1].*h.Position;
%             subplot(1,2,1); 
%             axis equal
%             pdeplot(thermalmodelT,'XYData',T(:,end),'Contour','on','ColorMap','hot'); 
%             axis equal
%             title 'Temperature, Final Time, Transient Solution'
%             subplot(1,2,2); 
%             axis equal
%             plot(tlist, T(nid,:)); 
%             hold on
%             plot(tlist, T(nid2,:));
%             grid on
%             title 'Temperature at Center as a Function of Time';
%             xlabel 'Time, seconds'
%             ylabel 'Temperature, degrees-Celsius'
% 
%            hold off
            centerT = T(nid,:);
            outT = T(nid2,:);
            Namecenter = sprintf("center_temp_%d%d%d", inittemp,risetemp,finaltemp)
            Nameouter = sprintf("outer_temp_%d%d%d", inittemp,risetemp,finaltemp)

            save(Namecenter,'centerT')
            save(Nameouter, 'outT')
        end
    end
end