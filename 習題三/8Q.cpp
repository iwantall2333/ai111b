#include <iostream>
using namespace std;

int p[8] ={0};//every row position
int count_=0;

bool checker(int nr,int nc){ //now_row, now_column
    for(int i = 0;i < nr;i++){
        if(p[i] == nc) return false; //column conflict
        if( (p[i] == (nc - (nr-i))) || (p[i] == (nc + (nr-i))) ) return false; // diagonal conflict
    }
    return true;
}

int dfs(int nr){
    if(nr == 8){    // print out solution
        count_++;
        cout<<"------------------------------------------\n";
        for(int j=0;j<8;j++){
            for(int i=0;i<p[j];i++){ cout<<"#"; }
            cout<<p[j];
            for(int i=p[j]+1;i<8;i++){ cout<<"#"; }
            cout<<"\n";
        }
    }

    for(int i=0;i<8;i++){   //traversal: row
        if(checker(nr,i)){
            p[nr] = i;      //put
            dfs(nr+1);      //next queen
        }
    }
    return 0;
}

int main(){
    dfs(0);
    cout<<"total:"<<count_<<" results";
    return 0;
}