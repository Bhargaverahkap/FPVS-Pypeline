function [outheader, outdata] = LW_norm_topos(header, data, selectedchannels, normMethod)

%
% Author :
% Corentin Jacques
% Institute of Neurosciences (IONS)
% Université catholique de louvain (UCL)
% Belgium
%

%transfer header to outheader
outheader=header;

%add history
i=length(outheader.history)+1;
outheader.history(i).description=['LW_norm_topos_',normMethod];
outheader.history(i).date=date;
outheader.datasize(2) = length(selectedchannels);

outdata = zeros(outheader.datasize);

normMethod = lower(normMethod);


%loop through all the data. 
for epochpos=1:size(data,1)    
        for indexpos=1:size(data,3)
            for dz=1:size(data,4)
                for dy=1:size(data,5);
                    switch normMethod
                        case 'mccarthy & wood'                            
                           scaling =  sqrt(sum(squeeze(data(epochpos,selectedchannels,indexpos,dz,dy,:)).^2,1));
%                            scaling =  sum(abs(squeeze(data(epochpos,selectedchannels,indexpos,dz,dy,:))),1);
                           disp(num2str(scaling));
                           outdata(epochpos,:,indexpos,dz,dy,:) = bsxfun(@rdivide,squeeze(data(epochpos,selectedchannels,indexpos,dz,dy,:)),scaling);
                            
                        case 'z-score';
                            outdata(epochpos,:,indexpos,dz,dy,:) = zscore(squeeze(data(epochpos,selectedchannels,indexpos,dz,dy,:)));
                            
                        otherwise
                            error('unknown normalize method!');                            
                    end
                end
            end
        end
end