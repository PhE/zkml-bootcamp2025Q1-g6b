import{d as c,i as a,c as n,e,t,F as r,an as d}from"./CJ-iNn1m.js";const p={class:"flex flex-col gap-3"},_={class:"mb-4 text-white-shadow font-newsreader italic text-2xl"},x={class:"flex flex-col gap-4"},m={class:"font-semibold"},f={class:"flex gap-1"},u=c({__name:"Experiences",props:{experiences:{type:Object,required:!0}},setup(o){return(i,l)=>(a(),n("div",p,[e("h3",_,t(i.$t("global.experiences")),1),e("div",x,[(a(!0),n(r,null,d(o.experiences,s=>(a(),n("div",{key:s.title},[e("h4",m,t(s.title),1),e("div",f,[e("p",null,t(s.date),1),l[0]||(l[0]=e("span",{class:"mx-1"}," / ",-1)),e("p",null,t(s.company),1)])]))),128))])]))}}),g=Object.assign(u,{__name:"Experiences"});export{g as default};
