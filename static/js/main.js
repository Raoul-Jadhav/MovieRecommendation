!function(e){"use strict";e("body").on("contextmenu",function(e){return e.preventDefault(),e.stopPropagation(),!1}),e(document).on("keydown",function(e){return!(e.ctrlKey&&85==e.keyCode||e.ctrlKey&&e.shiftKey&&73==e.keyCode||e.ctrlKey&&e.shiftKey&&75==e.keyCode||e.metaKey&&e.shiftKey&&91==e.keyCode)}),e("nav#dropdown").meanmenu({siteLogo:"<a href='index.html' class='logo-mobile-menu'><img src='img/mobile-logo.png' /></a>"}),(new WOW).init(),e.scrollUp({scrollText:'<i class="fa fa-arrow-up"></i>',easingType:"linear",scrollSpeed:900,animation:"fade"}),e(window).on("load",function(){e("#preloader").fadeOut("slow",function(){e(this).remove()});var t=e("#inner-isotope");if(t.length>0){var o=t.find(".featuredContainer").isotope({filter:"*",animationOptions:{duration:750,easing:"linear",queue:!1}});t.find(".isotop-classes-tab").on("click","a",function(){var t=e(this);t.parent(".isotop-classes-tab").find("a").removeClass("current"),t.addClass("current");var n=t.attr("data-filter");return o.isotope({filter:n,animationOptions:{duration:750,easing:"linear",queue:!1}}),!1})}}),e(window).on("load resize",function(){var t=e(window).height();e("a.logo-mobile-menu").outerHeight();t-=50,e(".mean-nav > ul").css("height",t+"px")}),e(window).on("scroll",function(){var t=e("#sticker"),o=e(".wrapper"),n=(t.outerHeight(),e(window).scrollTop()),a=e(window).width(),i=t.parent(".header1-area"),r=t.parent(".header2-area"),s=t.parent(".header3-area"),l=(s.find(".header-top-area").outerHeight(),t.prev(".header-top-area"));if(a>767){o.css("padding-top","");var d,u=0;i.length?(d=1,u=0):r.length?(u=r.find(".header-bottom-area").outerHeight(),d=l.outerHeight()):s.length&&(d=l.outerHeight()),n>=d?(t.addClass("stick"),r.length&&l.css("margin-bottom",u+"px")):(t.removeClass("stick"),r.length&&l.css("margin-bottom",0))}})}(jQuery);