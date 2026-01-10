
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import quad
import aplpy as ap
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits


# >>> basic function
def LN_tp(mu, sigma, tp):
    ln_tp = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(tp - mu)**2 / (2 * sigma**2))
    return ln_tp
def LN(x,mu,sigma):
    ln = (
            1 / ((np.pi * 2) ** 0.5 * sigma) * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
    )
    return ln
def PL(x, tp, ptp, index):
    pl = ptp * 10 ** (index * (x - tp))
    return pl
def npdf_model_log(x,mu,sigma,tp,index):
    # >>> log10: x, mu, sigma, tp
    # >>> normal: index

    # LN_tp_cdf = 0.5 * (1 + sp.special.erf((tp - mu) / (sigma * np.sqrt(2))))
    # f_total = LN_tp_cdf + LN_tp / (index * np.log(10))

    def LN_sub(x_sub):
        ln = (
                1 / ((np.pi * 2) ** 0.5 * sigma) * np.exp(-(x_sub - mu) ** 2 / 2 / sigma ** 2)
        )
        return ln

    def PL_sub(x_sub):
        ln_tp = LN_tp(mu, sigma, tp)
        pl = ln_tp * 10 ** (index * (x_sub - tp))
        return pl

    y = np.piecewise(
        x,
        [x <= tp, x > tp],
        [LN_sub, PL_sub]
    )

    return y


# >>> Bayes' formula
# for LN+PL
def log_likelihood(theta, x):
    mu,sigma,tp,index = theta
    pdf = npdf_model_log(x, mu, sigma, tp, index)
    f_total, f_total_err = quad(npdf_model_log,x.min(),x.max(),args = (mu, sigma, tp, index))
    if f_total == 0:
        f = 0.
    else:
        pdf = pdf / f_total
        pdf[pdf < 1e-50] = np.nan
        f = np.nansum(np.log(pdf))
    if np.isnan(f) == True:
        return 0.
    else:
        return f
def log_prior(theta, initial, tplimit,sigmalimit,mulimit,tp_prior):
    mu,sigma,tp,index = theta
    if tplimit is None:
        tplimit = [initial[2],0.3]
    else:
        pass
    if initial[0]-mulimit < mu < initial[0]+mulimit \
        and sigmalimit[0] < sigma < sigmalimit[1] \
        and tplimit[0] - tplimit[1] < tp < tplimit[0] + tplimit[1]\
        and np.arctan(-6.) < np.arctan(index) < np.arctan(-0.5) \
        and np.exp( -(tp - mu)**2 / (2*sigma**2) ) * (mu - tp) / (np.sqrt(2 * np.pi) * sigma**3) < LN_tp(mu,sigma,tp)*np.log(10)*index:
        if tp_prior is None:
            return 0.0
        else:
            tp_mu = tp_prior[0]
            tp_sigma = tp_prior[1]
            tp_A = tp_prior[2]
            return np.log(tp_A*1/((2*np.pi)**0.5 * tp_sigma) * np.exp(-(tp - tp_mu)**2 / (2 * tp_sigma**2)))
    else:
        return -np.inf
def log_probability(theta, x, initial, tplimit = None,sigmalimit = [0.05,0.5], mulimit = 0.05,tp_prior = None):
    lp = log_prior(theta, initial, tplimit, sigmalimit, mulimit,tp_prior)
    lk = log_likelihood(theta, x)
    return lp + lk
# for LN
def log_likelihood_LN(theta, x):
    mu,sigma = theta
    pdf = LN(x, mu, sigma)
    f_total, f_total_err = quad(LN,x.min(),x.max(),args = (mu, sigma))
    if f_total == 0:
        f = 0.
    else:
        pdf = pdf / f_total
        pdf = np.clip(pdf, 1e-50, None)  # progress 0 or negative value
        f = np.nansum(np.log(pdf))
    if np.isnan(f) == True:
        return 0.
    else:
        return f
def log_prior_LN(theta, initial,sigmalimit, mulimit):
    mu,sigma = theta

    if initial[0] - mulimit < mu < initial[0] + mulimit and sigmalimit[0] < sigma < sigmalimit[1]:
        return 0.0
    else:
        return -np.inf
def log_probability_LN(theta, x, initial,sigmalimit = [0.05,0.5], mulimit = 0.05):
    lp = log_prior_LN(theta, initial,sigmalimit, mulimit)
    return lp + log_likelihood_LN(theta, x)
# for PL
def log_likelihood_PL(theta, x, tp, ptp):
    index = theta
    pdf = PL(x, tp, ptp, index)
    f_total, f_total_err = quad(PL,x.min(),x.max(),args = (tp, ptp, index))
    if f_total == 0:
        f = 0.
    else:
        pdf = pdf / f_total
        pdf = np.clip(pdf, 1e-50, None)  # progress 0 or negative value
        f = np.nansum(np.log(pdf))
    if np.isnan(f) == True:
        return 0.
    else:
        return f
def log_prior_PL(theta):
    index = theta
    if np.arctan(-6.) < np.arctan(index) < np.arctan(-0.5):
        return 0.0
    else:
        return -np.inf
def log_probability_PL(theta, x, tp, ptp):
    lp = log_prior_PL(theta)
    return lp + log_likelihood_PL(theta, x, tp, ptp)


# >>> show and output N-PDF image; return den, bin; mark the fitting result if it's needed.
def npdf_show(sourcename,data,binwidth = 0.03,mcmcparapath = None,samplespath = None,yshift = None, show = True, vlines = None, vlabellist = None, pltstyle = "step1",
              hlines = None, hlabellist = None, savepath = None, plotcontrol = None, showlabel = "all", scale = "log10", Nmean = None, text = None):

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.family"] = "Times New Roman"

    data_1D = data.copy().ravel()
    data_1D = data_1D[~np.isnan(data_1D)]
    data_1D = data_1D[~np.isinf(data_1D)]
    if np.nanmax(data_1D) < 1e5:
        data_1D = 10**data_1D
    if scale == "log10":
        data_1D = np.log10(data_1D)
    elif scale == "eta":
        if Nmean is None:
            raise ValueError("Nmean must be specified if scale is eta")
        else:
            binwidth = binwidth * np.log(10)
            data_1D = np.log(data_1D / Nmean)

    bins = np.arange(np.nanmin(data_1D), np.nanmax(data_1D), binwidth)

    den, bin = np.histogram(data_1D, bins, density = False)
    bin = (bin[1:] + bin[:-1]) / 2
    den = den / len(data_1D)
    den = np.array(den)
    bin = np.array(bin)
    yerr = np.sqrt(den/len(data_1D))

    if show == True:
        fig = plt.figure(figsize = (7,5), dpi=120,facecolor="white")
        ax = fig.add_subplot()
        if pltstyle == "step1":
            ax.step(bin, den, where="mid", linewidth = 1., label = "N-PDF", zorder = 0)
            ax.errorbar(bin, den, yerr = yerr, fmt="None", capsize = 1.5,zorder = 1, linewidth = 0.8)
        elif pltstyle == "bar1":
            ax.step(bin, den, where="mid", linewidth = 0.7,color = "lightgrey", zorder = 0)
            ax.fill_between(bin,den,0, color = "lightgrey", label = "N-PDF",step = "mid", zorder = 0)
            ax.errorbar(bin, den, yerr = yerr, fmt="None",ecolor = "grey", capsize = 2.,zorder = 1, linewidth = 1.)
        if mcmcparapath is not None:
            mcmcpara = np.load(mcmcparapath)
            N0 = float(mcmcparapath.split("/")[-1].split("_")[2])
            N1 = float(mcmcparapath.split("/")[-1].split("_")[3])
            model = mcmcparapath.split("/")[-1].split("_")[4].split(".npy")[0]
            if model == "LNPL":
                Nthres = mcmcpara[2,0]
                Nthres_err1 = mcmcpara[2,1]
                Nthres_err2 = mcmcpara[2,2]
                mu = mcmcpara[0,0]
                sigma = mcmcpara[1,0]
                index = mcmcpara[3,0]
                if scale == "eta":
                    Nthres = np.log(10**Nthres/Nmean)
                    mu = np.log(10**mu/Nmean)
                    sigma = sigma * np.log(10)
                    N0 = np.log(10**N0/Nmean)
                    N1 = np.log(10**N1/Nmean)
                    index = index/np.log(10)
                    binwidth = binwidth/np.log(10)
                x_show = np.linspace(np.array([bin[0],N0-0.15]).max(), np.array([bin[-1],N1+0.5]).min(), 1000)
                model_int, model_int_err = quad(npdf_model_log, N0, N1, args=(mu,sigma,Nthres,index))
                mean_model = 1 / (N1 - N0) * model_int
                den1, bin1 = npdf_show(sourcename, data, binwidth, show=False, scale = scale, Nmean = Nmean)
                den1[np.where(bin1 < N0)[0]] = np.nan
                den1[np.where(bin1 > N1)[0]] = np.nan
                mean_data = np.nanmean(den1)
                scale1 = mean_data / mean_model
                if yshift is not None:
                    ax.plot(x_show, scale1*yshift*npdf_model_log(x_show, mu, sigma,Nthres,index),"r--", linewidth = 1.5, label = "Fitting result", zorder = 4)
                else:
                    ax.plot(x_show, scale1*npdf_model_log(x_show, mu, sigma,Nthres,index),"r--", linewidth = 1.5, label = "Fitting result", zorder = 4)
                if pltstyle == "step1":
                    line_Nthres = ax.axvline(Nthres, color="C1", linestyle="-.", linewidth = 1.2,
                                              label = r"$N_{\rm threshold}$" + r"=$%.1f\times10^{%i}{\rm cm^{-2}}$"%(10**Nthres/10**int(Nthres),Nthres), zorder = 3)
                    ax.fill_betweenx(np.linspace(0, 100, 1000), Nthres + Nthres_err1, Nthres + Nthres_err2, color="orange", alpha=0.3)
                elif pltstyle == "bar1":
                    line_Nthres = ax.axvline(Nthres, color="darkblue", linestyle="-", linewidth = 1.2,
                                              label = r"$N_{\rm threshold}$" + r"=$%.1f\times10^{%i}{\rm cm^{-2}}$"%(10**Nthres/10**int(Nthres),Nthres), zorder = 3)
                    ax.fill_betweenx(np.linspace(0, 100, 1000), Nthres + Nthres_err1, Nthres + Nthres_err2, color="lightblue", alpha=0.5)

                if samplespath is not None:
                    flat_samples = np.load(samplespath)
                    flat_samples = flat_samples[500:,:]
                    inds = np.random.randint(flat_samples.shape[0], size=100)
                    median = np.median(flat_samples,axis = 0)
                    factor = mcmcpara[:,0] / median
                    for ind in inds:
                        sample = flat_samples[ind]
                        sample = sample * factor
                        mu, sigma, Nthres, index = sample
                        if scale == "eta":
                            Nthres = np.log(10 ** Nthres / Nmean)
                            mu = np.log(10 ** mu / Nmean)
                            sigma = sigma * np.log(10)
                            index = index / np.log(10)
                        if yshift is not None:
                            ax.plot(x_show, scale1*yshift*npdf_model_log(x_show, mu, sigma,Nthres,index), "C1",linewidth = 0.5, alpha=0.08, zorder = 1)
                        else:
                            ax.plot(x_show, scale1*npdf_model_log(x_show, mu, sigma,Nthres,index), "C1",linewidth = 0.5,  alpha=0.08, zorder = 1)
            elif model == "PL":
                para1 = mcmcpara[:,0].tolist()
                Nthres = N0
                para = [N0,den[0],para1[0]]
                x_show = np.linspace(N0, np.array([bin[-1],N1+0.2]).min(), 1000)
                model_int, model_int_err = quad(PL, N0, N1, args=(para[0],para[1],para[2]))
                mean_model = 1 / (N1 - N0) * model_int
                den1, bin1 = npdf_show(sourcename, data, binwidth, show=False)
                den1[np.where(bin1 < N0)[0]] = np.nan
                den1[np.where(bin1 > N1)[0]] = np.nan
                mean_data = np.nanmean(den1)
                scale1 = mean_data / mean_model
                if yshift is not None:
                    ax.plot(x_show, scale1*yshift*PL(x_show, *para),"r--", linewidth = 1.5, label = "Fitting result", zorder = 4)
                else:
                    ax.plot(x_show, scale1*PL(x_show, *para),"r--", linewidth = 1.5, label = "Fitting result", zorder = 4)

                if pltstyle == "step1":
                    line_Nthres = ax.axvline(Nthres, color="C1", linestyle="-.", linewidth=1.2,
                                              label=r"$N_{\rm threshold}$" + r"=$%.1f\times10^{%i}{\rm cm^{-2}}$" % (
                                              10 ** Nthres / 10 ** int(Nthres), Nthres), zorder=3)
                elif pltstyle == "bar1":
                    line_Nthres = ax.axvline(Nthres, color="darkblue", linestyle="-", linewidth = 1.2,
                                              label = r"$N_{\rm threshold}$" + r"=$%.1f\times10^{%i}{\rm cm^{-2}}$"%(10**Nthres/10**int(Nthres),Nthres), zorder = 3)

                if samplespath is not None:
                    flat_samples = np.load(samplespath)
                    flat_samples = flat_samples[500:,:]
                    inds = np.random.randint(flat_samples.shape[0], size=100)
                    median = np.median(flat_samples,axis = 0)
                    factor = mcmcpara[:,0] / median
                    for ind in inds:
                        sample = flat_samples[ind]
                        sample = sample * factor
                        mu, sigma = sample
                        if scale == "eta":
                            mu = np.log(10 ** mu / Nmean)
                            sigma = sigma * np.log(10)
                        if yshift is not None:
                            ax.plot(x_show, scale1*yshift*LN(x_show, mu, sigma), "C1",linewidth = 0.5, alpha=0.08, zorder = 1)
                        else:
                            ax.plot(x_show, scale1*LN(x_show, mu, sigma), "C1",linewidth = 0.5,  alpha=0.08, zorder = 1)

            elif model == "LN":
                para = mcmcpara[:,0].tolist()
                mu = para[0]
                sigma = para[1]
                if scale == "eta":
                    mu = np.log(10**mu/Nmean)
                    sigma = sigma * np.log(10)
                    N0 = np.log(10**N0/Nmean)
                    N1 = np.log(10**N1/Nmean)
                    binwidth = binwidth/np.log(10)
                x_show = np.linspace(N0, np.array([bin[-1], N1 + 0.2]).min(), 1000)
                model_int, model_int_err = quad(LN, N0, N1, args=(mu,sigma))
                mean_model = 1 / (N1 - N0) * model_int
                den1, bin1 = npdf_show(sourcename, data, binwidth, show=False, scale = scale, Nmean = Nmean)
                den1[np.where(bin1 < N0)[0]] = np.nan
                den1[np.where(bin1 > N1)[0]] = np.nan
                mean_data = np.nanmean(den1)
                scale1 = mean_data / mean_model
                if yshift is not None:
                    ax.plot(x_show, yshift*scale1*LN(x_show, mu, sigma),"r--", linewidth = 1.6, label = "Fitting result", zorder = 4)
                else:
                    ax.plot(x_show, scale1*LN(x_show, mu, sigma),"r--", linewidth = 1.6, label = "Fitting result", zorder = 4)

                # error line
                if samplespath is not None:
                    flat_samples = np.load(samplespath)
                    flat_samples = flat_samples[500:,:]
                    inds = np.random.randint(flat_samples.shape[0], size=100)
                    median = np.median(flat_samples,axis = 0)
                    factor = mcmcpara[:,0] / median
                    for ind in inds:
                        sample = flat_samples[ind]
                        sample = sample * factor
                        mu, sigma = sample
                        if scale == "eta":
                            mu = np.log(10 ** mu / Nmean)
                            sigma = sigma * np.log(10)
                        if yshift is not None:
                            ax.plot(x_show, scale1*yshift*LN(x_show, mu, sigma), "C1",linewidth = 0.5, alpha=0.08, zorder = 1)
                        else:
                            ax.plot(x_show, scale1*LN(x_show, mu, sigma), "C1",linewidth = 0.5,  alpha=0.08, zorder = 1)

        else:
            pass
        if vlines is not None:
            for vlinei in range(len(vlines)):
                if vlabellist is not None:
                    ax.axvline(vlines[vlinei], linestyle="-.", linewidth=0.8, label = vlabellist[vlinei], zorder = 0)
                else:
                    ax.axvline(vlines[vlinei], linestyle="-.", linewidth=0.8, zorder = 0)
        if hlines is not None:
            for hlinei in range(len(hlines)):
                if hlabellist is not None:
                    ax.axhline(hlines[hlinei]/len(data_1D), linestyle="-",color = "C7", linewidth= 1., label = hlabellist[hlinei], zorder = 0)
                else:
                    ax.axhline(hlines[hlinei]/len(data_1D), linestyle="-",color = "C7", linewidth= 1., zorder = 0)
        ax.text(0.03,0.93,sourcename,horizontalalignment='left',verticalalignment='center',transform=ax.transAxes, fontsize = 16)

        if scale == "log10":
            ax.set_xlabel(r"${\rm log_{10}}(N_{\rm H_2})$", fontsize = 14)
        elif scale == "eta":
            ax.set_xlabel(r"$ln(N/<N>)$", fontsize = 14)
        ax.set_ylabel(r"Probability Distribution", fontsize = 14)
        if showlabel == "all":
            plt.legend(frameon = False, fontsize = 12, loc = 1)
        elif showlabel == "Nthres" and mcmcparapath is not None:
            plt.legend(handles = [line_Nthres],frameon = False, fontsize = 12, loc = 1)
        elif showlabel is False:
            pass

        if text is not None:
            for i in range(len(text)):
                text_x = text[i][0]
                text_y = text[i][1]
                text_t = text[i][2]
                fontsize = text[i][3]
                color = text[i][4]
                ax.text(text_x,text_y,text_t,transform=ax.transAxes, fontsize = fontsize, color = color)

        # plt.grid(linestyle = "--", linewidth = 0.6, alpha = 0.5)
        if plotcontrol is not None:
            xmin,xmax,ymin,ymax = plotcontrol
            if scale == "eta":
                xmin = np.log(10**xmin/Nmean)
                xmax = np.log(10**xmax/Nmean)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(10**ymin,10**ymax)
            ax_right = ax.twinx()
            ax_right.set_ylim(10**ymin * len(data_1D),10**ymax * len(data_1D))
            ax_right.set_ylabel(r"$N_{\rm pixel}$", fontsize = 14)
            ax_right.set_yscale("log")
        else:
            ax.set_ylim(np.array([den.max()/5e3,den.min()/2]).max(),den.max() * 3)
            ax_right = ax.twinx()
            ax_right.set_ylim(np.array([den.max()/5e3,den.min()/2]).max() * len(data_1D),den.max() * 3 * len(data_1D))
            ax_right.set_ylabel(r"$N_{\rm pixel}$", fontsize = 14)
            ax_right.set_yscale("log")
        ax.set_yscale("log")

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight", transparent = True, format = "pdf")
        else:
            pass
        plt.show()

    return den, bin


# >>> show the NH2 map together with the Nthres contour.
def map_show(sourcename,hdu,distance = None,contourlist = None,contour_color = ["C1","blue","red","green"],savepath = None, vmin = None, vmax = None, stretch = "linear"):

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.family"] = "Times New Roman"

    colors = [
    (0,"#000000"),
    (0.5,"#92A7B2"),
    (1,"#FFFFFF")
    ]
    cmap = LinearSegmentedColormap.from_list("mymap",colors)

    # preprocess
    data = hdu.data.copy()
    header = hdu.header.copy()
    data[np.isinf(data)] = np.nan
    if np.nanmax(data) > 1e5:
        data = np.log10(data)
    data[data < 20] = np.nan
    hdu = fits.PrimaryHDU(data, header = header)

    fig = ap.FITSFigure(hdu,dpi=100,facecolor = "white")
    fig.show_colorscale(cmap= "gist_yarg", pmin = 2.5, vmin = vmin,vmax = vmax,stretch = stretch)

    if distance is not None:
        scalelength = 1/distance / 2 / np.pi * 360
        fig.add_scalebar(length=scalelength)
        fig.scalebar.set_label('1 pc')
        fig.scalebar.set_font_size(20)
        fig.scalebar.set_color("white")
    else:
        pass

    # label sourcename
    ax = fig.ax
    ax.text(0.03, 0.04, sourcename,
            transform=ax.transAxes,
            fontsize=24,
            color='black',
            ha='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # plot contour (Nthres)
    if contourlist is not None:
        for contouri in range(len(contourlist)):
            ctr = contourlist[contouri]
            if ctr > 1e5:
                ctr = np.log10(ctr)
            fig.show_contour(hdu,levels = [ctr],colors = contour_color[contouri])

    # set up colorbar
    fig.add_colorbar()
    fig.colorbar.set_axis_label_text(r'$\mathrm{log_{10}(N_{H_2})}$ [$\mathrm{cm^{-2}}$]')
    fig.colorbar.set_axis_label_font(size=20)
    fig.colorbar.set_font(size = 16)
    fig.colorbar.set_pad(0.1)
    fig.colorbar.set_location('top')

    try:
        fig.add_beam()
        fig.beam.set_corner("top left")

    except:
        pass

    fig.tick_labels.set_font(size=20)
    fig.axis_labels.set_font(size=20)

    if savepath is not None:
        fig.savefig(savepath,dpi=300,format = 'pdf',transparent = True)


# >>> some tools
def M_calculate(map,distance,pixelsize,Nrange):
    '''
    distance should be in pc
    pixelsize should be in arcsec
    Values in Nrange should not be in logarithmic
    '''

    data = map.copy()   # H2 column density, in cm-2
    data[np.isinf(data)] = np.nan
    if np.nanmax(data) < 100:
        data = 10**data
        print("Warning: input data may be logarithmic, they have been converted to normal scale.")
    else:
        pass

    mu = 2.8
    m_H = 1.674 * 10 ** (-27)  # in kg
    solarmass = 1.98855e30  # in kg
    pc_to_cm = 3.0857e18

    data[data < Nrange[0]] = np.nan
    data[data > Nrange[1]] = np.nan

    Nsum = np.nansum(data)

    M = (distance * pixelsize / (3600*360) * 2 * np.pi * pc_to_cm)**2 * Nsum * mu * m_H / solarmass

    return M
def N_beam(data,beamsize,image = False):
    data1d = data.copy().ravel()
    data1d[np.isinf(data1d)] = np.nan
    data1d = data1d[~np.isnan(data1d)]
    if np.nanmax(data1d) > 1e8:
        data1d = np.log10(data1d)
    sorted_data = np.sort(data1d)
    cdf = np.linspace(0, 1, len(sorted_data))
    beamsize = beamsize / len(sorted_data)
    beamsize = 1-beamsize
    for i in range(len(sorted_data)):
        if cdf[i] > beamsize:
            N_beam = sorted_data[i]
            break
    if image == True:
        plt.figure(dpi=120, facecolor='w')

        plt.plot(sorted_data, cdf)
        plt.axhline(beamsize, color="grey", linewidth=0.8, label="Beamsize")
        plt.axvline(N_beam, color="C1", linestyle="--", linewidth=0.8)

        plt.legend(fontsize=12, frameon=False)
        plt.xlabel(r"${\rm log_{10}}(N_{\rm thres})$", fontsize=14)
        plt.ylabel(r"CDF", fontsize=14)
        plt.xlim(np.nanpercentile(data1d,75), np.nanmax(data1d) + 0.08)
        plt.ylim(0.8, 1.02)
        plt.show()
    else:
        pass
    return N_beam


'''
last revise: 2025.06.30

- add Npixel axis on the right
- update LN part in npdf_show
'''