function [out_header,out_data,message_string]=RLW_average_epochs(header,data,varargin);
%RLW_sum_epochs
%
%Sum epochs
%
%varargin
%'sum'
%

operation='sum';

%parse varagin
if isempty(varargin);
else
    %operation
    a=find(strcmpi(varargin,'operation'));
    if isempty(a);
    else
        operation=varargin{a+1};
    end;
end;

%init message_string
message_string={};

%prepare out_header
out_header=header;
%change number of epochs
out_header.datasize(1)=1;

%init out_data
out_data=zeros(out_header.datasize);

%perform the operation
switch operation
    case 'sum'
        out_data(1,:,:,:,:,:)=sum(data,1);
end;

%adjust events
if isfield(out_header,'events');
    for event_pos=1:length(out_header.events);
        out_header.events(event_pos).epoch=1;
    end;
end;

%delete epochdata
if isfield(out_header,'epochdata');
    rmfield(out_header,'epochdata');
end;