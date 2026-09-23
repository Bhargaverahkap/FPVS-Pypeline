function varargout = GLW_normalize_topographies(varargin)
% GLW_NORMALIZE_TOPOGRAPHIES M-file for GLW_normalize_topographies.fig
%      GLW_NORMALIZE_TOPOGRAPHIES, by itself, creates a new GLW_NORMALIZE_TOPOGRAPHIES or raises the existing
%      singleton*.
%
%      H = GLW_NORMALIZE_TOPOGRAPHIES returns the handle to a new GLW_NORMALIZE_TOPOGRAPHIES or the handle to
%      the existing singleton*.
%
%      GLW_NORMALIZE_TOPOGRAPHIES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GLW_NORMALIZE_TOPOGRAPHIES.M with the given input arguments.
%
%      GLW_NORMALIZE_TOPOGRAPHIES('Property','Value',...) creates a new GLW_NORMALIZE_TOPOGRAPHIES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GLW_normalize_topographies_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GLW_normalize_topographies_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GLW_normalize_topographies

% Last Modified by GUIDE v2.5 30-Sep-2014 08:57:36

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GLW_normalize_topographies_OpeningFcn, ...
                   'gui_OutputFcn',  @GLW_normalize_topographies_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GLW_normalize_topographies is made visible.
function GLW_normalize_topographies_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GLW_normalize_topographies (see VARARGIN)

% Choose default command line output for GLW_normalize_topographies
handles.output = hObject;


% Update handles structure
guidata(hObject, handles);

%fill listbox with inputfiles
inputfiles=varargin{2};
set(handles.filebox,'String',inputfiles);
set(handles.filebox,'Value',1);

%load header of first inputfile
st=get(handles.filebox,'String');
header=LW_load_header(st{1});
%fill listbox1
for i=1:header.datasize(2);
    stringlist{i}=header.chanlocs(i).labels;
end;
set(handles.listbox1,'String',stringlist);
set(handles.editlist,'Value',[]);


% UIWAIT makes GLW_normalize_topographies wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GLW_normalize_topographies_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in filebox.
function filebox_Callback(hObject, eventdata, handles)
% hObject    handle to filebox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns filebox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from filebox


% --- Executes during object creation, after setting all properties.
function filebox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filebox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in process_button.
function process_button_Callback(hObject, eventdata, handles)
% hObject    handle to process_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%Get filenames to process
inputfiles=get(handles.filebox,'String');

%Method.
methodVal = get(handles.method_popupmenu,'value');
methodString = get(handles.method_popupmenu,'string');
normMethod = methodString{methodVal};

%set selected channels.
selectedchannels = get(handles.editlist,'Value');

for filepos=1:length(inputfiles);
    
    %load header and data
    [header,data] = LW_load(inputfiles{filepos});
      
    %Normalize.
    [header, data] =  LW_norm_topos(header, data, selectedchannels, normMethod);
    
     prefixx = get(handles.prefixtext,'String');        
     LW_save(inputfiles{filepos},prefixx,header,data);
        
end
    
disp('*** Finished');

function prefixtext_Callback(hObject, eventdata, handles)
% hObject    handle to prefixtext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of prefixtext as text
%        str2double(get(hObject,'String')) returns contents of prefixtext as a double


% --- Executes during object creation, after setting all properties.
function prefixtext_CreateFcn(hObject, eventdata, handles)
% hObject    handle to prefixtext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Frequency_textbox_Callback(hObject, eventdata, handles)
% hObject    handle to Frequency_textbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Frequency_textbox as text
%        str2double(get(hObject,'String')) returns contents of Frequency_textbox as a double


% --- Executes during object creation, after setting all properties.
function Frequency_textbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Frequency_textbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to prefixtext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of prefixtext as text
%        str2double(get(hObject,'String')) returns contents of prefixtext as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to prefixtext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_addFreqBandName.
function checkbox_addFreqBandName_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_addFreqBandName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_addFreqBandName


% --- Executes on selection change in method_popupmenu.
function method_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to method_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns method_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from method_popupmenu


% --- Executes during object creation, after setting all properties.
function method_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to method_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in addbutton.
function addbutton_Callback(hObject, eventdata, handles)
% hObject    handle to addbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Insert
% Add selected channels in listbox1 at position selected in listbox2 (stored in editlist)
value1=get(handles.listbox1,'Value');
value2=get(handles.listbox2,'Value');
position=value2(1);
currentselection=get(handles.editlist,'Value');
if position==1;
    part1=[];
    part2=currentselection;
else
    part1=currentselection(1:position-1);
    part2=currentselection(position:length(currentselection));
end;
%concatenate
value2=[part1,value1,part2];
%update
set(handles.editlist,'Value',value2);
%update edit listbox2;
stringlist=get(handles.listbox1,'String');
set(handles.listbox2,'String',stringlist(value2));
%update selection 
set(handles.listbox1,'Value',[]);
set(handles.listbox2,'Value',[]);


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Delete
deleteselection=get(handles.listbox2,'Value');
currentselection=get(handles.editlist,'Value');
currentselection(deleteselection)=[];
set(handles.editlist,'Value',currentselection);
%update edit listbox2;
stringlist=get(handles.listbox1,'String');
set(handles.listbox2,'ListboxTop',1);
set(handles.listbox2,'String',stringlist(currentselection));
%update selection 
set(handles.listbox1,'Value',[]);
set(handles.listbox2,'Value',[]);

% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Add All
stringlist=get(handles.listbox1,'String');
set(handles.editlist,'Value',1:1:length(stringlist));
currentselection=get(handles.editlist,'Value');
%update edit listbox2;
set(handles.listbox2,'String',stringlist(currentselection));
%update selection 
set(handles.listbox1,'Value',[]);
set(handles.listbox2,'Value',[]);


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Delete All
set(handles.editlist,'Value',[]);
currentselection=[];
set(handles.listbox2,'ListboxTop',1);
stringlist=get(handles.listbox1,'String');
set(handles.listbox2,'String',stringlist(currentselection));
%update selection 
set(handles.listbox1,'Value',[]);
set(handles.listbox2,'Value',[]);



function editlist_Callback(hObject, eventdata, handles)
% hObject    handle to editlist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editlist as text
%        str2double(get(hObject,'String')) returns contents of editlist as a double


% --- Executes during object creation, after setting all properties.
function editlist_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editlist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


