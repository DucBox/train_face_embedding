// Validate the JavaScript embedded in a generated viewer: decisions, persistence,
// approval blocking, whitelist CSV exports. Run with a generated index.html path.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const page=fs.readFileSync(process.argv[2],'utf8');
const script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
const manifest=JSON.parse(fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[2]),'manifest.json'),'utf8'));
const storage=new Map(),downloads=[],blobs=new Map(),alerts=[];
const cards=new Map(manifest.cases.map(c=>[c.id,{id:c.id,dataset:{},querySelector:()=>({textContent:''})}]));
const sandbox={
 Blob, alert:m=>alerts.push(m), setTimeout:fn=>fn(),
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
 URL:{createObjectURL:b=>{const k='blob:'+blobs.size;blobs.set(k,b);return k;},revokeObjectURL:()=>{}},
 document:{getElementById:id=>id==='counts'?{textContent:''}:cards.get(id),
   querySelectorAll:()=>[...cards.values()],createElement:()=>({click(){downloads.push({name:this.download,blob:blobs.get(this.href)});}})},
};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const neg=manifest.cases.find(c=>c.kind==='negative'&&c.a===0);
const pos=manifest.cases.find(c=>c.kind==='positive'&&c.errors.length===0);
const bad=manifest.cases.find(c=>c.errors.length>0);
const pending=manifest.cases.find(c=>c.kind==='negative'&&c.id!==neg.id);
assert.ok(neg&&pos&&bad&&pending,'Fixture must include valid and invalid cases');
sandbox.decide(neg.id,'keep');sandbox.decide(pos.id,'keep');sandbox.decide(bad.id,'keep');
assert.equal(alerts.length,1);assert.equal(cards.get(bad.id).dataset.status,'pending');
sandbox.exportFile('approved_identity_pairs.csv');sandbox.exportFile('approved_images.csv');
sandbox.exportFile('excluded_images.csv');sandbox.exportFile('review_decisions.csv');
(async()=>{
 const texts=await Promise.all(downloads.map(d=>d.blob.text()));
 assert.equal(texts[0].trim().split('\r\n').length,2,'Only approved pair is exported');
 assert.ok(texts[0].includes('"0"'),'Class zero is preserved');
 assert.equal(texts[1].trim().split('\r\n').length,2,'Only approved image is exported');
 assert.equal(texts[2].trim().split('\r\n').length,1,'Pending/error cases are not silently marked excluded');
 assert.ok(texts[3].includes('"pending"'),'Pending status recorded');
 sandbox.decide(pos.id,'unsure');sandbox.exportFile('excluded_images.csv');
 const excluded=await downloads.at(-1).blob.text();assert.ok(excluded.includes('"unsure"'));
 // New page context, same browser storage: decisions survive navigation.
 const next={...sandbox};vm.createContext(next);vm.runInContext(script,next);
 assert.equal(cards.get(neg.id).dataset.status,'keep');
 assert.equal(cards.get(pos.id).dataset.status,'unsure');
 console.log('PASS: exports, pending/unsure handling, label-error approval block and cross-page persistence.');
})().catch(e=>{console.error(e);process.exitCode=1;});
