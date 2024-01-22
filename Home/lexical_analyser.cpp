#include<iostream>
#include<set>
#include<ifstream>
#include<algorithm>

using namespace std;

int main()
{
ifstream infile;
infile.fopen("/home/ravi/program.txt","w");
string keywords={"for","while","int","char","double","return","main"};
char symbols={'&','*','^','%',';'};
set<string>key_found;
set<string>symbol_found;
set<string>identifier;
set<string>symbol_found;


//infile.open("program.txt",ios::in);
string words;
char ch,word;

int p=0;
while(ch.getchar(infile)!=eof)
{
if(symbols.find(element) != symbols.end(););
symbol_found.push(data);

}

cout<<"Identifiers:-\n";
for(int i=0;i<identifier.size();i++)
cout<<identifier[i]<<" ";

cout<<"Keywords Found:-\n";
for(int i=0;i<key_found.size();i++)
cout<<key_found[i]<<" ";

cout<<"Symbols found:-\n";
for(int i=0;i<symbol_found.size();i++)
cout<<symbol_found[i]<<" ";


return 0;
}
