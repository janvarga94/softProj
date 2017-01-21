from Models import *
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

class VideoProcessor:

    def __init__(self):
        self.linija = None
        self.sum = 0
        self.prethodniBoxoviSlike = []
        self.maxRastojanjeBrojeva = 120

        pass

    def SetLiniju(self,linija):
        self.linija = linija

    def IsLinijaSet(self):
        if self.linija == None:
            return False
        else:
            return True

    def Process(self, boxoviSlike):
        if self.linija == None:
            raise Exception('Linija nije postavljena')
        if len(self.prethodniBoxoviSlike) == 0:
            self.prethodniBoxoviSlike.append(boxoviSlike)
            return
        for i in range(10):
            boxoviZaJedanBroj = boxoviSlike[i]
            if len(boxoviZaJedanBroj) == 0:
                continue

            self.proveriPresekZaBroj_i(boxoviZaJedanBroj,i)



        self.prethodniBoxoviSlike.append(boxoviSlike)

    def proveriPresekZaBroj_i(self,boxoviZaJedanBroj,i):
        for referentniBox in boxoviZaJedanBroj:

            najblizi = None
            rastojanje = 8000
            lenPrethodnih = len(self.prethodniBoxoviSlike)
            for ii in range(lenPrethodnih):
                if rastojanje > self.maxRastojanjeBrojeva and ii < 4: #gledam maximalno 4 koraka unazad
                    najblizi, rastojanje = self.__pronadjiNajblizi(referentniBox,self.prethodniBoxoviSlike[lenPrethodnih - ii - 1][i],i)
                else:
                    break



            if rastojanje < self.maxRastojanjeBrojeva: #znaci to je isti broj samo pomeren u ovom frejmu, sledece proveravamo dal linija sece putanju ovog broja
                #debug
                r,c = self.__srednjaVrednostRedaKolone(referentniBox)
                r1, c1 = self.__srednjaVrednostRedaKolone(najblizi)
                plt.plot([c,c1],[r,r1], 'g')
             #   plt.plot([c1], [r1], 'g>')
                plt.plot([self.linija.col1], [self.linija.row1], 'bo')
                plt.plot([self.linija.col2], [self.linija.row2], 'bo')
                #end
                (rowNajblizi,colNajblizi) = self.__srednjaVrednostRedaKolone(najblizi)
                (rowReferentni, colReferentni) = self.__srednjaVrednostRedaKolone(referentniBox)
                daliSeceLiniju = self.__intersect(TackaXY(colNajblizi,rowNajblizi), TackaXY(colReferentni,rowReferentni), TackaXY(self.linija.col1,self.linija.row1), TackaXY(self.linija.col2,self.linija.row2))
                if daliSeceLiniju:
                    print("sece : {0}".format(i))
                    self.sum += i

    # Return true if line segments AB and CD intersect
    def __intersect(self,A, B, C, D):
        return self.__ccw(A, C, D) != self.__ccw(B, C, D) and self.__ccw(A, B, C) != self.__ccw(A, B, D)

    def __ccw(self,A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    def __distance(self,row1,col1,row2,col2):
        return sqrt( (row2 - row1)**2 + (col2 - col1)**2 )

    def __sortirajPoRedu(self,pozicije):
        return  sorted(pozicije, key=lambda pozicija: (pozicija.row1 + pozicija.row2)/2)

    def __pronadjiNajblizi(self, referentni, brojeviPozicije,ind):
        if len(brojeviPozicije) == 0:
            return (None, 8000)
        rowReferentnog, colReferentnog = self.__srednjaVrednostRedaKolone(referentni)

        indexNajblizeg = -1
        rastojanjeNajblizeg = 8000 #neka velika vrednost, trazimo sto manju
        for i in range(0,len(brojeviPozicije)):
            rowTrenutnog, colTrenutnog = self.__srednjaVrednostRedaKolone(brojeviPozicije[i])
            if rowTrenutnog > rowReferentnog or colTrenutnog > colReferentnog: #jer se brojevi krecu desno dole
                continue
            rastojanje = self.__distance(rowReferentnog,colReferentnog,rowTrenutnog,colTrenutnog)
            if rastojanje < rastojanjeNajblizeg:
                indexNajblizeg = i
                rastojanjeNajblizeg = rastojanje
        
        return brojeviPozicije[indexNajblizeg], rastojanjeNajblizeg

    def __srednjaVrednostRedaKolone(self,pozicija):
        return ((pozicija.row1 + pozicija.row2)/2,(pozicija.col1 + pozicija.col2)/2)

__name__ = "VideoProcessor"