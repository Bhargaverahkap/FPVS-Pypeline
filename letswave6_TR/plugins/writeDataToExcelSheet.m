


function written = writeDataToExcelSheet(fileNm, sheetNm, dataToWrite, xlswriter)
% appends matrix of data into excel sheet: takes left corner of first free
% line. (column A; first free row = left upper corner of data). If
% sheet/file does not exist yet, it will be created.

if nargin < 2
    if ischar(fileNm)
        written = initialize(fileNm);
    else  %  when giving only xlswriter to the function
        fileNm.Workbook.Close;
        fileNm.Excel.Quit;
        fileNm.Excel.delete;
%         h = actxGetRunningServer('Excel.Application');
%         h.Quit
%         h.delete
        written = 'ok';
    end
else
    if nargin < 4
        xlswriter = initialize(fileNm);
    end
    
    sheetNm = char(sheetNm);
    Excel = xlswriter.Excel;
    Workbook = xlswriter.Workbook;
    
    Sheets = Workbook.Sheets;
    % write behavioral output of trial (1 line)
    try
        Sht = get(Sheets, 'Item', sheetNm);
        % Select range in which the data have to be written
        fstNewLine = size(Sht.usedRange.value, 1) +1;
    catch
        Sht = Workbook.Sheets.Add;
        Sht.Name = sheetNm;
        fstNewLine = 1;
        % try to delete the standard sheets
        try
            Sheets.Item('Sheet1').Delete;
            Sheets.Item('Sheet2').Delete;
            Sheets.Item('Sheet3').Delete;
        catch
        end
    end
    
    [m,n] = size(dataToWrite);
    lastrow = num2str(fstNewLine+m-1);   % Construct last row as a string.
    firstrow = num2str(fstNewLine);      % Convert first row to string image.
    lastcol = dec2base27(base27dec('A')+n-1); % Construct last column.
    range = ['A' firstrow ':' lastcol lastrow]; % Final range string.
    
    % change values in range in sheet by dataToWrite
    Sht.Range(range).Value = dataToWrite;
    % Workbook.SaveAs(fileNm);
    Workbook.Save;
    %     Workbook.Close;
    
    
    %     Excel.Quit;
    %     Excel.delete;
    if nargin < 4
        fileNm.Workbook.Close;
        fileNm.Excel.Quit;
        fileNm.Excel.delete;
        written = 'ok';
    end
    
end

%------------------------------------------------------------------------------
function xlswriter = initialize(fileNm)
fileNm = char(fileNm);

xlswriter.Excel = actxserver('Excel.Application');

% If file and sheets already exist
try
    xlswriter.Workbook = xlswriter.Excel.Workbooks.Open(fileNm);
    % If file and sheets have to be created.
catch
    xlswriter.Workbook = xlswriter.Excel.Workbooks.Add;
    xlswriter.Workbook.SaveAs(fileNm);
end

%------------------------------------------------------------------------------
function s = dec2base27(d)

%   DEC2BASE27(D) returns the representation of D as a string in base 27,
%   expressed as 'A'..'Z', 'AA','AB'...'AZ', until 'IV'. Note, there is no zero
%   digit, so strictly we have hybrid base26, base27 number system.  D must be a
%   negative integer bigger than 0 and smaller than 2^52, which is the maximum
%   number of columns in an Excel worksheet.
%
%   Examples
%       dec2base(1) returns 'A'
%       dec2base(26) returns 'Z'
%       dec2base(27) returns 'AA'
%-----------------------------------------------------------------------------
b = 26;
symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

d = d(:);
if d ~= floor(d) | any(d < 0) | any(d > 1/eps)
    error('MATLAB:xlswrite:Dec2BaseInput',...
        'D must be an integer, 0 <= D <= 2^52.');
end

% find the number of columns in new base
n = max(1,round(log2(max(d)+1)/log2(b)));
while any(b.^n <= d)
    n = n + 1;
end

% set b^0 column
s(:,n) = rem(d,b);
while n > 1 && any(d)
    if s(:,n) == 0
        s(:,n) = b;
    end
    if d > b
        % after the carry-over to the b^(n+1) column
        if s(:,n) == b
            % for the b^n digit at b, set b^(n+1) digit to b
            s(:,n-1) = floor(d/b)-1;
        else
            % set the b^(n+1) digit to the new value after the last carry-over.
            s(:,n-1) = rem(floor(d/b),b);
        end
    else
        s(:,n-1) = []; % remove b^(n+1) digit.
    end
    n = n - 1;
end
s = symbols(s);
%------------------------------------------------------------------------------
function d = base27dec(s)
%   BASE27DEC(S) returns the decimal of string S which represents a number in
%   base 27, expressed as 'A'..'Z', 'AA','AB'...'AZ', until 'IV'. Note, there is
%   no zero so strictly we have hybrid base26, base27 number system.
%
%   Examples
%       base27dec('A') returns 1
%       base27dec('Z') returns 26
%       base27dec('IV') returns 256
%-----------------------------------------------------------------------------

d = 0;
b = 26;
n = numel(s);
for i = n:-1:1
    d = d+(s(i)-'A'+1)*(b.^(n-i));
end
%-------------------------------------------------------------------------------




