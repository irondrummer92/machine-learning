from random import randint

# Defining a tic tac toe game. It has a tic tac toe board, 2 players who either place X or O
# The match is won when either X or O places 3 tokes in a row
# The game object accepts a move from each player and places it in the location specified,
# checks for winning state. If won, it returns the winning player name. Else, it continues
# with the next round till winning state is obtained

class tictactoeGame:
    
    def __init__(self, board, player1, player2):
        self.board = board
        self.player1 = player1
        self.player2 = player2
        self.winState = 0
        self.winConditions = [
            # Horizontal Win states
            (0,1,2),(3,4,5),(6,7,8),
            # Vertical Win States
            (0,3,6),(1,4,7),(2,5,8),
            # Diagonal Win States
            (0,4,8),(2,4,6)
        ]
        self.winner = None
        
    def runGame(self):
        if(self.player1.playerType == "X"):
    
            while(not(self.winstate == 1)):
                move1 = self.player1.chooseMove(self.board)
                self.board.update(self.player1.playerType, move1)
                self.winState,self.winner = self.checkWinState()
                move2 = self.player2.chooseMove(self.board)
                self.board.update(self.player1.playerType, move2)
                self.winState,self.winner = self.checkWinState()
        
        else:
            
            while(not(self.winstate == 1)):
                move2 = self.player2.chooseMove(self.board)
                self.board.update(self.player1.playerType, move2)
                self.winState,self.winner = self.checkWinState()
                move1 = self.player1.chooseMove(self.board)
                self.board.update(self.player1.playerType, move1)
                self.winState,self.winner = self.checkWinState()
        
        print "Game Ends...!"
        
        if(self.winner == None):
            print "Its a draw..!"
        else:
            print self.winner.name + " Wins... !"
        
    # Checking for winning state involves checking the state at the end of every move
    # 3 consecutive X or Os either horizontally, diagonally or vertically ends game
    def checkWinState(self):
        # Setting empty locations for Xs and Os
        
        XLocations = self.board.getPlayerLocations("X")
        OLocaitons = self.board.getPlayerLocations("O")

        for condition in self.winConditions:
            if(all(x in XLocations for x in condition)):
                self.winState = 1
                if(self.player1.playerType == "X"):
                    self.winner = self.player1
                else:
                    self.winner = self.player2
            elif(all(x in OLocations for x in condition)):
                self.winState = 1
                if(self.player2.playerType == "O"):
                    self.winner = self.player2
                else:
                    self.winner = self.player1
            return
        
        if(self.winState == 0):
            if(self.board.isEmpty()):
                return
            else:
                self.winState = 1
                self.Winner = None
                return
                

# Defining a tic tac toe board
class tictactoeBoard:
    
    def __init__(self):
        self.boxes = 9
        self.boxInputs = ["" for i in range(8)]
        
    def update(self, playerType, move):
        self.boxInputs[move] = playerType 
    
    def isEmpty(self):
        if(all(x == "" for x in self.board.boxInputs)):
            return True
        else:
            return False
        
    def getPlayerLocations(self, playerType):
        if(self.isEmpty):
            return []
        else:
            playerLocations = []
            for i in range(8):
                if self.boxInputs[i] == playerType:
                    playerLocations.append(i)
            return playerLocations
# Defining a tic tac toe player. The player chooses its moves and places an X or O depending on the type of player
class tictactoePlayer:
    
    def __init__(self, playerType, name = None):
        
        if(not(len(playerType) > 1) and not(playerType not in ("X","O"))):
            self.playerType = playerType
        else: 
            raise Exception("Not a valid type for the player. Enter either X or O")
            
        if(not(name == None)):
            self.name = name
        else:
            self.name = "Player " + self.playerType
        
    def chooseMove(self,boardState):
        print "Choosing Move"
        XLocations = boardState.getXLocations("X")
        OLocations = boardState.getOLocations("O")
               

#class autoTictactoPlayer(tictactoePlayer):
    
    # Lot of combinations possible for an auto player. Askin
        
#class learningTictactoePlayer(tictactoePlayer):
#    def chooseMove(self,boardState):
#        print "This player learns how to play tic-tac-toe. Patience...!"
